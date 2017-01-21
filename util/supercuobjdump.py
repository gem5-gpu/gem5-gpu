#!/usr/bin/env python

import optparse
import os
import re
from subprocess import *
import sys

def which(program):
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def cleanEntryLine(line):
    start = line.find('.entry ') + len('.entry ')
    end = line.find(' ', start)
    return line[start:end]

def cleanVisibleFuncLine(line):
    # Process lines of the form:
    #  .visible .func _ZN14ComponentSpace6isBossEj
    #  .visible .func (.param .s32 __cudaretf__ZN14ComponentSpace6isBossEj) _ZN14ComponentSpace6isBossEj
    start = line.find('.visible .func ') + len('.visible .func ')
    nextstart = line.find(') ', start)
    if nextstart > 0:
        start = nextstart + 2
    end = line.find(' ', start)
    return line[start:end]

def cleanFuncLine(line):
    # Process lines of the form:
    #  .func _ZN14ComponentSpace6isBossEj
    #  .func (.param .s32 __cudaretf__ZN14ComponentSpace6isBossEj) _ZN14ComponentSpace6isBossEj
    start = line.find('.func ') + len('.func ')
    nextstart = line.find(') ', start)
    if nextstart > 0:
        start = nextstart + 2
    end = line.find(' ', start)
    return line[start:end]

parser = optparse.OptionParser()
parser.add_option("--debug", action="store_true", default=False, help="Debug print all lines")
parser.add_option("--decimal", action="store_true", default=False, help="Print PCs in decimal")
(options, args) = parser.parse_args()

if len(args) < 1:
    print >>sys.stderr, 'ERROR: Must specify CUDA binary'
    sys.exit(0)

cubin = args[0]
if not os.path.exists(cubin):
    print >>sys.stderr, 'ERROR: CUDA binary (%s) does not exist' % cubin
    sys.exit(0)

if which('cuobjdump') is None:
    print >>sys.stderr, 'ERROR: Must have cuobjdump in PATH variable'
    sys.exit(0)

process = Popen("cuobjdump -ptx %s" % cubin, shell=True, stdout=PIPE)
output = process.communicate()[0]

during_function = False
during_section = False
last_func_line = ''
pc = 0

for line in output.split('\n'):
    if options.debug:
        print 'DEBUG: %s' % line
    line = line.replace('\t','    ')
    end = line.find('//')
    if end < 0:
        end = len(line)
    line = line[0:end]
    line = line.rstrip()

    if '{' in line and '{%' not in line and '{_' not in line:
        if during_function or during_section:
            print line
            assert(0)
        if '.section ' in line:
            print 'SECTION: ',
            during_section = True
        else:
            if '.entry' in last_func_line:
                print 'FUNCTION: %s' % cleanEntryLine(last_func_line)
            elif '.visible .func' in last_func_line:
                print 'FUNCTION: %s' % cleanVisibleFuncLine(last_func_line)
            elif '.func' in last_func_line:
                print 'FUNCTION: %s' % cleanFuncLine(last_func_line)
            else:
                print >>sys.stderr, 'ERROR: Strange function declaration'
                sys.exit(0)
            during_function = True

    if '}' in line and '{%' not in line and '{_' not in line:
        assert (during_function and not during_section) or (not during_function and during_section)
        print '}'
        print ''
        during_function = False
        during_section = False

    if during_function:
        if ('{' in line and '{%' not in line and '{%' not in line) or line == '':
            print line.lstrip()
        elif ':' in line or '.reg' in line or '.loc ' in line or '.local ' in line or '.shared ' in line or '.param ' in line or '.pragma' in line:
            print '        %s' % line.lstrip()
        else:
            if line[-1] != ';':
                print line
                assert(0)
            if options.decimal:
                pcstring = '%u:' % pc
            else:
                pcstring = '0x%x:' % pc
            print '%-10s%s' % (pcstring, line)
            pc += 8
    elif during_section:
        print '        %s' % line.lstrip()
    else:
        if '.func' in line or '.entry' in line:
            last_func_line = line

