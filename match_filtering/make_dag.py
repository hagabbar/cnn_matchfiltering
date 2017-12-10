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
    Nsig=sys.argv[3] # number of signals in each run

    ifo='H1'
    #segs=['1','2','3','4','5','6','7','8','9','10']
    segs=['10']
    ts_num = ['0','1']

    jobtypes=['0','1']

    count = 0
    #fdag = open("snr8.dag",'w')
    fdag = open("snr%s.dag" % segs[0],'w')
    for idx, seg in enumerate(segs):
        for jobtype in jobtypes:
            for i in range(0,10000,100):
                count = count + 1
                data = '/home/hunter.gabbard/glasgow/github_repo_code/cnn_matchfiltering/data/BBH_testing_1s_8192Hz_10Ksamp_25n_iSNR%s_Hdet_astromass_1seed_ts_%s.sav' % (seg,jobtype) 
                params = '/home/hunter.gabbard/glasgow/github_repo_code/cnn_matchfiltering/data/BBH_testing_1s_8192Hz_10Ksamp_25n_iSNR%s_Hdet_astromass_1seed_params_%s.sav' % (seg,jobtype)
                base = '%sfull_bank_snr%s/ts_%s/' % (b,seg,jobtype)
                add_job(fdag, jobtype, count, tb=tb, d=data, b=base, Nsig=Nsig, st=i, params=params)

