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

# Functions to check all the constraints
# CHECK 1 : If local dir has given some object, remote dir must know about that object.
def local_to_remote_check
  #checking locals are consistent with remotes
  obj_ptr_re = /\[\d{1},(.*)\]/
  locRW_re = /\locRW\:(.?)\,/
  recalled_re = /recalled\:(.?)\,/
  count = 0
  $arr_local_store.each do |local_file|
    local_file.each do |line|
      unless line.chomp.empty?
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
    count = count + 1
  end
end

#CHECK 2: If obj in local dir and has a reqsRW, then remote must have contented for that obj.
def local_reqsRW_remote_contended
  obj_ptr_re = /\[\d{1},(.*)\]/
  reqsRW_re = /reqsRW:\<(.*)\>/
  contended_re = /contended\:(.?)\,/
  count = 0
  $arr_local_store.each do |local_file|
    local_file.each do |line|
      unless line.chomp.empty?
        obj_ptr = line.match(obj_ptr_re)
        reqsRW = line.match(reqsRW_re)
        reqs_arr = reqsRW[1].split(/,/)
        reqs_arr.size.times do |i|
          found = false
          $arr_remote_store[reqs_arr[i].to_i].each do |r_line|
            if r_line.include? obj_ptr[1]
              found = true
              contended = r_line.match(contended_re)
              if contended[1].to_i == 0
                p "OMG! #{count} has received a request, for object #{obj_ptr[1]} from #{reqs_arr[i].to_i}, but its not conteneded there"
              end
            end
          end
          if !found and !count.eql?reqs_arr[i].to_i
              p "OMG! #{count} has received a request, for object #{obj_ptr[1]} from #{reqs_arr[i].to_i}, but remote dir at this host doesn't know"
            end
        end
      end
    end
    count = count + 1
  end
end

#CHECK 3: Obj is present locally, its not contented, and there is request for it, but its not given.
def not_contended_with_reqs
  obj_ptr_re = /\[\d{1},(.*)\]/
  reqsRW_re = /reqsRW:\<(.*)\>/
  locRW_re = /\locRW\:(.?)\,/
  contended_re = /contended\:(.?)\,/
  count = 0
  $arr_local_store.each do |local_file|
    local_file.each do |line|
      unless line.chomp.empty?
        obj_ptr = line.match(obj_ptr_re)
        lockRW = line.match(locRW_re)
        reqsRW = line.match(reqsRW_re)
        reqs_arr = reqsRW[1].split(/,/)
        contended = line.match(contended_re)
        #if lockRW is empty, it should be contended locally
        if lockRW[1].eql?"" and reqs_arr.size > 0
          if contended[1].to_i == 0
            p "OMG! #{count} has an object #{obj_ptr} , which is needed by remote hosts #{reqs_arr} and is not locally contended"
          end
        end
      end
    end
    count = count + 1
  end
end

options = OptParserClass.parse(ARGV)
check_opts(options)
p options.t.class
construct_fileNames(options)
#
######constraint_checking
local_to_remote_check
local_reqsRW_remote_contended
not_contended_with_reqs
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
