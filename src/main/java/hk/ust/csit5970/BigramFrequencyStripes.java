package hk.ust.csit5970;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.regex.Pattern;

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
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

public class BigramFrequencyStripes extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(BigramFrequencyStripes.class);
    private static final Pattern WORD_BOUNDARY = Pattern.compile("\\s*\\b\\s*");

    private static class MyMapper extends
            Mapper<LongWritable, Text, Text, HashMapStringIntWritable> {
        private static final Text KEY = new Text();
        private static final HashMapStringIntWritable STRIPE = new HashMapStringIntWritable();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString().toLowerCase();
            String[] words = WORD_BOUNDARY.split(line);

            if (words.length < 2) return;

            for (int i = 0; i < words.length - 1; i++) {
                String current = words[i].replaceAll("[^a-zA-Z]", "");
                String next = words[i+1].replaceAll("[^a-zA-Z]", "");
                
                if (current.isEmpty() || next.isEmpty()) continue;
                
                STRIPE.clear();
                STRIPE.increment(next);
                KEY.set(current);
                context.write(KEY, STRIPE);
            }
        }
    }

    private static class MyCombiner extends
            Reducer<Text, HashMapStringIntWritable, Text, HashMapStringIntWritable> {
        private final static HashMapStringIntWritable SUM_STRIPES = new HashMapStringIntWritable();

        @Override
        public void reduce(Text key, Iterable<HashMapStringIntWritable> stripes, 
                Context context) throws IOException, InterruptedException {
            SUM_STRIPES.clear();
            for (HashMapStringIntWritable stripe : stripes) {
                SUM_STRIPES.plus(stripe);
            }
            context.write(key, SUM_STRIPES);
        }
    }

    private static class MyReducer extends
            Reducer<Text, HashMapStringIntWritable, PairOfStrings, FloatWritable> {
        private final static HashMapStringIntWritable SUM_STRIPES = new HashMapStringIntWritable();
        private final static PairOfStrings BIGRAM = new PairOfStrings();
        private final static FloatWritable FREQ = new FloatWritable();

        @Override
        public void reduce(Text key, Iterable<HashMapStringIntWritable> stripes, 
                Context context) throws IOException, InterruptedException {
            SUM_STRIPES.clear();
            int total = 0;
            
            for (HashMapStringIntWritable stripe : stripes) {
                for (Map.Entry<String, Integer> entry : stripe.entrySet()) {
                    String word = entry.getKey();
                    int count = entry.getValue();
                    SUM_STRIPES.increment(word, count);
                    total += count;
                }
            }

            BIGRAM.set(key.toString(), "");
            FREQ.set(total);
            context.write(BIGRAM, FREQ);

            for (Map.Entry<String, Integer> entry : SUM_STRIPES.entrySet()) {
                BIGRAM.set(key.toString(), entry.getKey());
                FREQ.set(entry.getValue() / (float)total);
                context.write(BIGRAM, FREQ);
            }
        }
    }

    public BigramFrequencyStripes() {}

    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    @SuppressWarnings("static-access")
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
            System.err.println("Error parsing command line: " + exp.getMessage());
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
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? 
                Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        Configuration conf = getConf();
        Job job = Job.getInstance(conf);
        job.setJobName(BigramFrequencyStripes.class.getSimpleName());
        job.setJarByClass(BigramFrequencyStripes.class);
        job.setNumReduceTasks(reduceTasks);

        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(HashMapStringIntWritable.class);
        job.setOutputKeyClass(PairOfStrings.class);
        job.setOutputValueClass(FloatWritable.class);

        job.setMapperClass(MyMapper.class);
        job.setCombinerClass(MyCombiner.class);
        job.setReducerClass(MyReducer.class);

        Path outputDir = new Path(outputPath);
        FileSystem.get(conf).delete(outputDir, true);

        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return 0;
    }

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new BigramFrequencyStripes(), args);
    }
}
