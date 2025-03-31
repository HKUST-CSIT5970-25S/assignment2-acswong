package hk.ust.csit5970;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

/**
 * Compute the bigram relative frequencies using "pairs" approach
 */
public class BigramFrequencyPairs extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(BigramFrequencyPairs.class);

    private static class MyMapper extends
            Mapper<LongWritable, Text, PairOfStrings, IntWritable> {

        private static final IntWritable ONE = new IntWritable(1);
        private static final PairOfStrings BIGRAM = new PairOfStrings();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            String[] words = line.trim().split("\\s+");
            
            for (int i = 0; i < words.length - 1; i++) {
                if (words[i].length() > 0 && words[i+1].length() > 0) {
                    BIGRAM.set(words[i], words[i + 1]);
                    context.write(BIGRAM, ONE);
                }
            }
        }
    }

    private static class MyReducer extends
            Reducer<PairOfStrings, IntWritable, PairOfStrings, FloatWritable> {

        private final static FloatWritable VALUE = new FloatWritable();
        private String lastLeftWord = null;
        private int marginalCount = 0;
        private Map<String, Integer> bigramCounts = new HashMap<String, Integer>();

        @Override
        public void reduce(PairOfStrings key, Iterable<IntWritable> values,
                Context context) throws IOException, InterruptedException {
            String leftWord = key.getLeftElement();
            String rightWord = key.getRightElement();
            
            // When switching to new left word, output previous results
            if (lastLeftWord != null && !leftWord.equals(lastLeftWord)) {
                // Output marginal count
                VALUE.set(marginalCount);
                context.write(new PairOfStrings(lastLeftWord, ""), VALUE);
                
                // Output relative frequencies
                for (Map.Entry<String, Integer> entry : bigramCounts.entrySet()) {
                    float frequency = (float)entry.getValue() / marginalCount;
                    VALUE.set(frequency);
                    context.write(new PairOfStrings(lastLeftWord, entry.getKey()), VALUE);
                }
                
                // Reset for new left word
                bigramCounts.clear();
                marginalCount = 0;
            }
            
            // Calculate sum for current bigram
            int sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }
            
            bigramCounts.put(rightWord, sum);
            marginalCount += sum;
            lastLeftWord = leftWord;
        }
        
        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            if (lastLeftWord != null) {
                // Output last marginal count
                VALUE.set(marginalCount);
                context.write(new PairOfStrings(lastLeftWord, ""), VALUE);
                
                // Output last set of relative frequencies
                for (Map.Entry<String, Integer> entry : bigramCounts.entrySet()) {
                    float frequency = (float)entry.getValue() / marginalCount;
                    VALUE.set(frequency);
                    context.write(new PairOfStrings(lastLeftWord, entry.getKey()), VALUE);
                }
            }
        }
    }
    
    private static class MyCombiner extends
            Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
        private static final IntWritable SUM = new IntWritable();

        @Override
        public void reduce(PairOfStrings key, Iterable<IntWritable> values,
                Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }
            SUM.set(sum);
            context.write(key, SUM);
        }
    }

    private static class MyPartitioner extends
            Partitioner<PairOfStrings, IntWritable> {
        @Override
        public int getPartition(PairOfStrings key, IntWritable value,
                int numReduceTasks) {
            return (key.getLeftElement().hashCode() & Integer.MAX_VALUE)
                    % numReduceTasks;
        }
    }

    public BigramFrequencyPairs() {
    }

    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    @SuppressWarnings({ "static-access" })
    public int run(String[] args) throws Exception {
        Options options = new Options();

        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("input path").create(INPUT));
        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("output path").create(OUTPUT));
        options.addOption(OptionBuilder.withArgName("num").hasArg()
                .withDescription("number of reducers").create(NUM_REDUCERS));

        CommandLine cmdline;
        CommandLineParser parser = new GnuParser();

        try {
            cmdline = parser.parse(options, args);
        } catch (ParseException exp) {
            System.err.println("Error parsing command line: "
                    + exp.getMessage());
            return -1;
        }

        if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
            System.out.println("args: " + Arrays.toString(args));
            HelpFormatter formatter = new HelpFormatter();
            formatter.setWidth(120);
            formatter.printHelp(this.getClass().getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        String inputPath = cmdline.getOptionValue(INPUT);
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
                .parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        LOG.info("Tool: " + BigramFrequencyPairs.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        Configuration conf = getConf();
        Job job = Job.getInstance(conf);
        job.setJobName(BigramFrequencyPairs.class.getSimpleName());
        job.setJarByClass(BigramFrequencyPairs.class);

        job.setNumReduceTasks(reduceTasks);

        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        job.setMapOutputKeyClass(PairOfStrings.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(PairOfStrings.class);
        job.setOutputValueClass(FloatWritable.class);

        job.setMapperClass(MyMapper.class);
        job.setCombinerClass(MyCombiner.class);
        job.setPartitionerClass(MyPartitioner.class);
        job.setReducerClass(MyReducer.class);

        Path outputDir = new Path(outputPath);
        FileSystem.get(conf).delete(outputDir, true);

        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime)
                / 1000.0 + " seconds");

        return 0;
    }

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new BigramFrequencyPairs(), args);
    }
}
