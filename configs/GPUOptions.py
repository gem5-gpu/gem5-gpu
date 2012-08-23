
import os

import m5


def addGPUOptions(parser):
    parser.add_option("--clusters", default=16, help="Number of shader core clusters in the gpu that GPGPU-sim is simulating", type="int")
    parser.add_option("--cores_per_cluster", default=1, help="Number of shader cores per cluster in the gpu that GPGPU-sim is simulating", type="int")
    parser.add_option("--sc_l1_size", default="64kB", help="size of l1 cache hooked up to each sc")
    parser.add_option("--sc_l2_size", default="1MB", help="size of L2 cache divided by num L2 caches")
    parser.add_option("--sc_l1_assoc", default=4, help="associativity of l1 cache hooked up to each sc", type="int")
    parser.add_option("--sc_l2_assoc", default=16, help="associativity of L2 cache backing SC L1's", type="int")
    parser.add_option("--shMemDelay", default=1, help="delay to access shared memory in gpgpu-sim ticks", type="int")
    parser.add_option("--kernel_stats", default=False, action="store_true", help="Dump statistics on GPU kernel boundaries")
    parser.add_option("--total-mem-size", default='2GB', help="Total size of memory in system")
    parser.add_option("--gpu_l1_buf_depth", type="int", default=16, help="Number of buffered L1 requests per shader")

def parseGpgpusimConfig(options):
    # parse gpgpu config file
    # First check the cwd, and if there is not a gpgpusim.config file there
    # Use the template found in gem5-fusion/configs/gpu_config and fill in
    # the missing information with command line options.
    usingTemplate = False
    gpgpusimconfig = os.path.join(os.getcwd(), 'gpgpusim.config')
    if not os.path.isfile(gpgpusimconfig):
        gpgpusimconfig = os.path.join(os.path.dirname(__file__), 'gpu_config/gpgpusim.config.template')
        usingTemplate = True
        if not os.path.isfile(gpgpusimconfig):
            print >>sys.stderr, "Unable to find gpgpusim config (%s)" % gpgpusimconfig
            sys.exit(1)
    f = open(gpgpusimconfig, 'r')
    config = f.read()
    f.close()

    if usingTemplate:
        print "Using template and command line options for gpgpusim.config"
        config = config.replace("%clusters%", str(options.clusters))
        config = config.replace("%cores_per_cluster%", str(options.cores_per_cluster))
        config = config.replace("%icnt_file%", os.path.join(os.path.dirname(__file__), "gpu_config/icnt_config_fermi_islip.txt"))
        options.num_sc = options.clusters*options.cores_per_cluster
        f = open(m5.options.outdir+'/gpgpusim.config', 'w')
        f.write(config)
        f.close()
        gpgpusimconfig = m5.options.outdir+'/gpgpusim.config'
    else:
        print "Using gpgpusim.config for clusters and cores_per_cluster"
        start = config.find("-gpgpu_n_clusters ")+len("-gpgpu_n_clusters ")
        end = config.find('-', start)
        gpgpu_n_clusters = int(config[start:end])
        start = config.find("-gpgpu_n_cores_per_cluster ")+len("-gpgpu_n_cores_per_cluster ")
        end = config.find('-', start)
        gpgpu_n_cores_per_cluster = int(config[start:end])
        num_sc = gpgpu_n_clusters*gpgpu_n_cores_per_cluster
        options.num_sc = num_sc

    return gpgpusimconfig

def addMemCtrlOptions(parser):
    parser.add_option("--mem_ctl_latency", type="int", default=-1, help="Memory controller latency in cycles")
    parser.add_option("--mem_freq", type="string", default=None, help="Memory controller frequency")
    parser.add_option("--membus_busy_cycles", type="int", default=-1, help="Memory bus busy cycles per data transfer")
    parser.add_option("--membank_busy_time", type="string", default=None, help="Memory bank busy time in ns (CL+tRP+tRCD+CAS)")

def setMemoryControlOptions(system, options):
    from m5.params import Latency
    for i in xrange(options.num_dirs):
        cntrl = eval("system.dir_cntrl%d" % i)
        if options.mem_freq:
            cntrl.memBuffer.clock = options.mem_freq
        if options.mem_ctl_latency >= 0:
            cntrl.memBuffer.mem_ctl_latency = options.mem_ctl_latency
        if options.membus_busy_cycles > 0:
            cntrl.memBuffer.basic_bus_busy_time = options.membus_busy_cycles
        if options.membank_busy_time:
            mem_cycle_seconds = float(cntrl.memBuffer.clock.period)
            bank_latency_seconds = Latency(options.membank_busy_time)
            cntrl.memBuffer.bank_busy_time = long(bank_latency_seconds.period / mem_cycle_seconds)
