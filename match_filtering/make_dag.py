#! /usr/bin/env python
import sys

from gwpy.segments import DataQualityFlag

def add_job(the_file, job_type, job_number, **kwargs):
    job_id="%s%.6u" % (job_type, job_number)
    the_file.write("JOB %s target.sub\n" % (job_id))
    vars_line=" ".join(['%s="%s"'%(arg,str(val))
                            for arg,val in kwargs.iteritems()])
    the_file.write("VARS %s %s\n" % (job_id, vars_line))
    the_file.write("\n")

if __name__ == "__main__":
    tb=sys.argv[1] # full path to template bank
    b=sys.argv[2] # full path to output

    ifo='H1'
    segs=['1','2','3','4','5','6','7','8','9','10']
    ts_num = ['0','1']

    jobtypes=['0','1']

    fdag = open("my.dag",'w')
    for idx, seg in enumerate(segs):
        for jobtype in jobtypes:
            data = '/home/hunter.gabbard/glasgow/github_repo_code/cnn_matchfiltering/data/BBH_testing_1s_8192Hz_10Ksamp_25n_iSNR%s_Hdet_astromass_1seed_ts_%s.sav' % (seg,jobtype) 
            base = '%sfull_bank_snr%s/ts_%s' % (b,seg,jobtype)
            add_job(fdag, jobtype, idx, tb=tb, d=data, b=base)

