import sys

req_version = (2,7)
cur_version = (sys.version_info[0],sys.version_info[1])

if cur_version == req_version:
   from run import program
   program()
else:
   print "Python v2.7 is required to run this package"