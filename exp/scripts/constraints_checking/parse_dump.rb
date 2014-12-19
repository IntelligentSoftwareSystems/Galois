# ##############################
# Parsing dump files
#
# Author : Gurbinder Gill
# email  : gill@cs.utexas.edu
# Date   : 17 Dec, 2014
#
# ##############################


require 'optparse'
require 'pp'
require 'ostruct'

#Gloabal Variables
$arr_local_store = []
$arr_remote_store = []

class OptParserClass
  def self.parse(args)
    options = OpenStruct.new
    options.n = 0;
    options.t = 0;


    opt_parser = OptionParser.new do |opts|
      opts.banner = "Usage: script.rb [options]"

      opts.on("-n", "--n_hosts hosts", Integer, "Give number of hosts") do |hosts|
        options.n = hosts
      end

      opts.on("-t", "--TimeStamp timestamp", Integer, "Which time stamp u want to process.") do |timestamp|
        options.t = timestamp
      end

      opts.on_tail('-h', '--help', String, 'Display Help.') do
        puts opts
        exit
      end
    end
  opt_parser.parse!(args)
  options
  end #end pasrse

end #OptParserClass

def check_opts(opts)
  if opts.n <= 1
    print "Number of hosts must be > 1 : specify : "
    opts.n = gets.chomp
    print "\n"
  end

  if opts.t == 0
    print "Using Default timestamp 0\n"
  end

end #check_opts

def construct_fileNames(opts)
  t = opts.t

  #open and store local files
  opts.n.times do |i|
    #arr_name = "local_#{i}"
    $arr_local_store[i] = open_files("dump_local_#{i}_#{t}.txt")
    #puts arr_local_store[i][0]
    #puts "------------------------------\n"
  end

  #checking
  #opts.n.times do |i|
    #puts $arr_local_store[i][0]
    #puts "------------------------------------\n"
  #end

  #checking
  #arr_local_store[0].each do |line|
    #puts line
  #end

  #open and store remote files
  opts.n.times do |i|
    $arr_remote_store[i] = open_files("dump_remote_#{i}_#{t}.txt")
    #puts arr_remote_store[i][0]
    #puts "------------------------------\n"
  end
end #construct_fileNames

def open_files(filename)
  file = File.new(filename, "r")
  array = []
  while (line = file.gets)
    array << line.chomp
  end
  file.close
  return array
end #open_files

# Main program to check all the constraints
def constraint_checking
  #checking locals are consistent with remotes
  obj_ptr_re = /\[\d{1},(.*)\]/
  locRW_re = /\locRW\:(.?)\,/
  recalled_re = /recalled\:(.?)\,/
  count = 0
  $arr_local_store.each do |local_file|
    local_file.each do |line|
      obj_ptr = line.match(obj_ptr_re)
      remote_host = line.match(locRW_re)
      recalled_for = line.match(recalled_re)

      #check if remote host knows about this obj_ptr
      if !remote_host[1].eql?"" and recalled_for[1].eql?""
        found = false
        $arr_remote_store[remote_host[1].to_i].each do |r_line|
          if r_line.include? obj_ptr[1]
            found = true
          end
        end
        if !found 
          p "OMG! #{count} gave its object #{obj_ptr[1]} to #{remote_host[1]}, but it doesn't seem to know about it"
        end

      end
    end
  end
end

options = OptParserClass.parse(ARGV)
check_opts(options)
p options.t.class
construct_fileNames(options)
constraint_checking

#host_0_local =
##
#
#
#my_re = /\[\d{1},(.*)\]/
#m = my_re.match(line)
#p m[1]
#locRW_re = /\locRW\:(.?)\,/
#
#string.include? "pattrn"
