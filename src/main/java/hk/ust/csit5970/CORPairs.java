package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.DoubleWritable;
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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

/**
 * Compute the correlation coefficients of bigrams using "pairs" approach
 */
public class CORPairs extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(CORPairs.class);

    /*
     * Done First-pass Mapper
     */
    private static class CORMapper1 extends Mapper<LongWritable, Text, Text, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private Text word = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            HashMap<String, Integer> word_set = new HashMap<String, Integer>();
            String clean_doc = value.toString().replaceAll("[^a-z A-Z]", " ");
            StringTokenizer doc_tokenizer = new StringTokenizer(clean_doc);
            
            while (doc_tokenizer.hasMoreTokens()) {
                String token = doc_tokenizer.nextToken();
                if (token.length() > 0) {
                    word.set(token);
                    context.write(word, ONE);
                }
            }
        }
    }

    /*
     * First-pass Reducer to Sum up word freq
     */
    private static class CORReducer1 extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    /*
     * Second-pass Mapper
     */
    public static class CORPairsMapper2 extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private PairOfStrings pair = new PairOfStrings();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer doc_tokenizer = new StringTokenizer(value.toString().replaceAll("[^a-z A-Z]", " "));
            HashSet<String> uniqueWords = new HashSet<String>();
            
            while (doc_tokenizer.hasMoreTokens()) {
                String word = doc_tokenizer.nextToken();
                if (word.length() > 0) {
                    uniqueWords.add(word);
                }
            }
            
            List<String> words = new ArrayList<String>(uniqueWords);
            Collections.sort(words);
            for (int i = 0; i < words.size(); i++) {
                for (int j = i + 1; j < words.size(); j++) {
                    pair.set(words.get(i), words.get(j));
                    context.write(pair, ONE);
                }
            }
        }
    }

    /*
     * Second-pass Combiner
     */
    private static class CORPairsCombiner2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
        private IntWritable sum = new IntWritable();

        @Override
        protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int total = 0;
            for (IntWritable val : values) {
                total += val.get();
            }
            sum.set(total);
            context.write(key, sum);
        }
    }

    /*
     * Second-pass Reducer which is compute COR(A, B)
     */
    public static class CORPairsReducer2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, DoubleWritable> {
        private final static Map<String, Integer> word_total_map = new HashMap<String, Integer>();
        private DoubleWritable result = new DoubleWritable();

        @Override
        protected void setup(Context context) throws IOException arbitrations, InterruptedException {
            Path middle_result_path = new Path("mid/part-r-00000");
            Configuration middle_conf = new Configuration();
            try {
                FileSystem fs = FileSystem.get(URI.create(middle_result_path.toString()), middle_conf);
                if (!fs.exists(middle_result_path)) {
                    throw new IOException(middle_result_path.toString() + " not exist!");
                }
                FSDataInputStream in = fs.open(middle_result_path);
                InputStreamReader inStream = new InputStreamReader(in);
                BufferedReader reader = new BufferedReader(inStream);

                String line;
                while ((line = reader.readLine()) != null) {
                    String[] line_terms = line.split("\t");
                    word_total_map.put(line_terms[0], Integer.valueOf(line_terms[1]));
                }
                reader.close();
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }

        @Override
        protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            String word1 = key.getLeftElement();
            String word2 = key.getRightElement();
            
            int pairCount = 0;
            for (IntWritable val : values) {
                pairCount += val.get();
            }
            
            Integer freqA = word_total_map.get(word1);
            Integer freqB = word_total_map.get(word2);
            
            if (freqA != null && freqB != null && freqA > 0 && freqB > 0) {
                double cor = (double) pairCount / (freqA * freqB);
                result.set(cor);
                context.write(key, result);
            }
        }
    }

    /*
     * partitioner: Partition based on left word
     */
    private static class MyPartitioner extends Partitioner<PairOfStrings, IntWritable> {
        @Override
        public int getPartition(PairOfStrings key, IntWritable value, int numReduceTasks) {
            return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
        }
    }

    /**
     * Creates an instance of this tool.
     */
    public CORPairs() {
    }

    private static final String INPUT = "input";
    private static final String MIDDLE = "middle";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    /**
     * Runs this tool with two MapReduce jobs.
     */
    @SuppressWarnings({"static-access"})
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
        String middlePath = "mid";
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        LOG.info("Tool: " + CORPairs.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        // First Job: Compute word frequencies
        Configuration conf1 = new Configuration();
        Job job1 = Job.getInstance(conf1, "CORPairs-FirstPass");
        job1.setJarByClass(CORPairs.class);
        job1.setMapperClass(CORMapper1.class);
        job1.setCombinerClass(CORReducer1.class);  // Add combiner for efficiency
        job1.setReducerClass(CORReducer1.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);
        FileInputFormat.setInputPaths(job1, new Path(inputPath));
        FileOutputFormat.setOutputPath(job1, new Path(middlePath));

        Path middleDir = new Path(middlePath);
        FileSystem.get(conf1).delete(middleDir, true);

        long startTime = System.currentTimeMillis();
        boolean success = job1.waitForCompletion(true);
        LOG.info("Job 1 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
        if (!success) return 1;

        // Second Job: Compute correlation coefficients
        Configuration conf2 = new Configuration();
        Job job2 = Job.getInstance(conf2, "CORPairs-SecondPass");
        job2.setJarByClass(CORPairs.class);
        job2.setMapperClass(CORPairsMapper2.class);
        job2.setCombinerClass(CORPairsCombiner2.class);
        job2.setPartitionerClass(MyPartitioner.class);
        job2.setReducerClass(CORPairsReducer2.class);
        job2.setMapOutputKeyClass(PairOfStrings.class);
        job2.setMapOutputValueClass(IntWritable.class);
        job2.setOutputKeyClass(PairOfStrings.class);
        job2.setOutputValueClass(DoubleWritable.class);
        job2.setNumReduceTasks(reduceTasks);
        FileInputFormat.setInputPaths(job2, new Path(inputPath));
        FileOutputFormat.setOutputPath(job2, new Path(outputPath));

        Path outputDir = new Path(outputPath);
        FileSystem.get(conf2).delete(outputDir, true);

        startTime = System.currentTimeMillis();
        success = job2.waitForCompletion(true);
        LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return success ? 0 : 1;
    }

    /**
     * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new CORPairs(), args);
    }
}
