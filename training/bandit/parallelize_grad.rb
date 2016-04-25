#!/usr/bin/env ruby

require 'trollop'

def usage
  STDERR.write "Usage: "
  STDERR.write "ruby parallelize_grad.rb -c <bandit.ini> [-e <epochs=10>] [--randomize/-z] [--reshard/-y] -s <#shards|0> [-p <at once=9999>] -i <input> -r <refs> [--qsub/-q] [--bandit_binary <path to bandit binary>] [-l \"l2 select_k 100000\"] [--extra_qsub \"-l virtual_free=24G\"]\n"
  exit 1
end

opts = Trollop::options do
  opt :lplp_args, "arguments for lplp.rb", :type => :string, :default => "l2 sum"
  opt :randomize, "randomize shards before each epoch", :type => :bool, :short => '-z', :default => false
  opt :reshard, "reshard after each epoch", :type => :bool, :short => '-y', :default => false
  opt :shards, "number of shards", :type => :int
  opt :processes_at_once, "have this number (max) running at the same time", :type => :int, :default => 9999
  opt :input, "input", :type => :string
  opt :references, "references", :type => :string
  opt :qsub, "use qsub", :type => :bool, :default => false
  opt :bandit_binary, "path to bandit binary", :type => :string
  opt :extra_qsub, "extra qsub args", :type => :string, :default => ""
  opt :per_shard_decoder_configs, "give special decoder config per shard", :type => :string, :short => '-o'
  opt :first_input_weights, "input weights for first iter", :type => :string, :default => '', :short => '-w'
  opt :global_dir, "directory for global data", :type => :string, :short => '-g'
end
usage if not opts[:config]&&opts[:shards]&&opts[:input]&&opts[:references]

bandit_dir = File.expand_path File.dirname(__FILE__)
if not opts[:bandit_binary]
  bandit_bin = "#{bandit_dir}/bandit_mt_grad"
else
  bandit_bin = opts[:bandit_binary]
end
ruby       = '/usr/bin/ruby'
lplp_rb    = "#{bandit_dir}/lplp.rb" #modified for summing
lplp_args  = opts[:lplp_args]
cat        = '/bin/cat'

ini        = opts[:config]
epochs     = 1
rand       = opts[:randomize]
reshard    = opts[:reshard]
predefined_shards = false
per_shard_decoder_configs = false
if opts[:shards] == 0
  predefined_shards = true
  num_shards = 0
  per_shard_decoder_configs = true if opts[:per_shard_decoder_configs]
else
  num_shards = opts[:shards]
end
input = opts[:input]
refs  = opts[:references]
use_qsub       = opts[:qsub]
shards_at_once = opts[:processes_at_once]
first_input_weights  = opts[:first_input_weights]
opts[:extra_qsub] = "-l #{opts[:extra_qsub]}" if opts[:extra_qsub]!=""
if not opts[:global_dir]
  dir = Dir.pwd
else
  dir = opts[:global_dir]
end

`mkdir #{dir}/work`

def make_shards(input, refs, num_shards, epoch, rand, dir)
  lc = `wc -l #{input}`.split.first.to_i
  index = (0..lc-1).to_a
  index.reverse!
  index.shuffle! if rand
  shard_sz = (lc / num_shards.to_f).round 0
  puts "shard size #{shard_sz}"
  leftover = lc - (num_shards*shard_sz)
  puts "leftover #{leftover}"
  leftover = 0 if leftover < 0
  in_f = File.new input, 'r'
  in_lines = in_f.readlines
  refs_f = File.new refs, 'r'
  refs_lines = refs_f.readlines
  shard_in_files = []
  shard_refs_files = []
  in_fns = []
  refs_fns = []
  new_num_shards = 0
  0.upto(num_shards-1) { |shard|
    break if index.size==0
    new_num_shards += 1
    in_fn = "#{dir}/work/shard.#{shard}.#{epoch}.in"
    shard_in = File.new in_fn, 'w+'
    in_fns << in_fn
    refs_fn = "#{dir}/work/shard.#{shard}.#{epoch}.refs"
    shard_refs = File.new refs_fn, 'w+'
    refs_fns << refs_fn
    0.upto(shard_sz-1) { |i|
      break if index.size==0
#      puts "line for shard i #{i}"
      j = index.pop
#      puts "line of input j  #{j}"
      shard_in.write in_lines[j]
      #shard_refs.write refs_lines[j] #don't split references
    }
    shard_refs.write refs_lines.join() #write full references to each shard
    shard_in_files << shard_in
    shard_refs_files << shard_refs
  }
  while leftover > 0
    j = index.pop
    shard_in_files[-1].write in_lines[j]
    #shard_refs_files[-1].write refs_lines[j]
    leftover -= 1
  end
  (shard_in_files + shard_refs_files).each do |f| f.close end
  in_f.close
  refs_f.close
  return in_fns, refs_fns, new_num_shards
end


input_files = []
refs_files = []
if predefined_shards
  input_files = File.new(input).readlines.map {|i| i.strip }
  refs_files = File.new(refs).readlines.map {|i| i.strip }
  if per_shard_decoder_configs
    decoder_configs = File.new(opts[:per_shard_decoder_configs]).readlines.map {|i| i.strip}
  end
  num_shards = input_files.size
else
  input_files, refs_files, num_shards = make_shards input, refs, num_shards, 0, rand, dir
end

0.upto(epochs-1) { |epoch|
  puts "epoch #{epoch+1}"
  pids = []
  input_weights = ''
  if epoch > 0 then input_weights = "--weights #{dir}/work/weights.#{epoch-1}" end
  weights_files = []
  shard = 0
  remaining_shards = num_shards
  while remaining_shards > 0
    shards_at_once.times {
      break if remaining_shards==0
      qsub_str_start = qsub_str_end = ''
      local_end = ''
      if use_qsub
        qsub_str_start = "qsub #{opts[:extra_qsub]} -l h_vmem=8G -wd /scratch/kreutzer/ -sync y -b y -j y -o #{dir}/work/out.#{shard}.#{epoch} -N bandit.#{shard}.#{epoch} \""
        qsub_str_end = "\""
        local_end = ''
      else
        local_end = "2>#{dir}/work/out.#{shard}.#{epoch}"
      end
      if per_shard_decoder_configs
        cdec_cfg = "--decoder_config #{decoder_configs[shard]}"
      else
        cdec_cfg = ""
      end
      if first_input_weights!='' && epoch == 0
        input_weights = "--weights #{first_input_weights}"
        puts "first input weights #{first_input_weights}"
      end
      puts "call #{bandit_bin} -c #{ini} #{cdec_cfg} #{input_weights} --input #{input_files[shard]} --reference #{refs_files[shard]} --weights_dump #{dir}/work/weights.#{shard}.#{epoch}#{qsub_str_end} #{local_end}"
      pids << Kernel.fork {
        `#{qsub_str_start}#{bandit_bin} -c #{ini} #{cdec_cfg} #{input_weights}\
          --input #{input_files[shard]}\
          --reference #{refs_files[shard]}\
          --weights_dump #{dir}/work/weights.#{shard}.#{epoch}#{qsub_str_end} #{local_end}`
      }
      weights_files << "#{dir}/work/weights.#{shard}.#{epoch}"
      shard += 1
      remaining_shards -= 1
    }
    puts "pids #{pids}"
    pids.each { |pid| Process.wait(pid) }
    pids.clear
  end
  `#{cat} #{dir}/work/weights.*.#{epoch}.txt > #{dir}/work/weights_cat`
  `#{ruby} #{lplp_rb} #{lplp_args} #{num_shards} < #{dir}/work/weights_cat > #{dir}/work/weights.#{epoch}`
  if rand and reshard and epoch+1!=epochs
    input_files, refs_files, num_shards = make_shards input, refs, num_shards, epoch+1, rand
  end
}

`rm #{dir}/work/weights_cat`

