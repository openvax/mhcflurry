#! /usr/bin/python

"""
GOAL:
=====
Compare two consecutive peptides in the cross-validation data sets.

If they are from different cv-partitions:
    1) Check whether they are similar.
    2) If similar, print out warning.


NOTE:
Of course, this is a quick/crude check. Not all pairs of peptides are compared.

Expectations:
=============
1) cv_gs data sets should not have any similar peptides.
2) Compared to cv_rnd, cv_sr should have much smaller number of similar peptides.

Execute Test:
=============
nosetests quick-check-cv-data.py

Will return 'OK' if everything passes.

"""


def get_peptide(line):
    delimiter = '\t'
    index_mhc = 1
    index_plen = 2
    index_cv = 3
    index_peptide = 4
    index_ineq = 5
    index_meas = 6
    row = line.strip().split(delimiter)
    cv = row[index_cv].strip()
    peptide = row[index_peptide].strip()
    mhc = row[index_mhc].strip()
    plen = row[index_plen].strip()
    ineq = row[index_ineq].strip()
    meas_ic50 = row[index_meas].strip()
    return (peptide, cv, mhc, plen, ineq, meas_ic50)

def seq_identity(pepa, pepb):
    """
    Two peptides will be considered similar if 80% sequence identity is observed.
    """
    assert len(pepa) == len(pepb)
    num_identity = sum([1 for (a, b) in zip(pepa, pepb) if a == b])
    fraction_identity = float(num_identity) / len(pepa)
    # print 'fraction_identity', fraction_identity, pepa, pepb
    return fraction_identity

def is_similar(pepa, pepb, sim_cutoff=0.80):
    """
    """
    is_sim = False
    s = seq_identity(pepa, pepb)
    if s >= sim_cutoff:
        is_sim = True
    # print 'is_similar', is_sim, s, pepa, pepb
    return is_sim

def count_peptides_similar(fname_cv, debug=False):
    # fname_bdata = sys.argv[1]
    print('DEBUG INPUT cross-validation data set =', fname_cv)

    lines = open(fname_cv, 'r').readlines()

    count_sim_peptides = 0
    for i in range(1, len(lines) - 1):
        line_a = lines[i]
        line_b = lines[i + 1]
        (pep_a, cv_a, mhc_a, plen_a, ineq_a, meas_ic50_a) = get_peptide(line_a)
        (pep_b, cv_b, mhc_b, plen_b, ineq_b, meas_ic50_b) = get_peptide(line_b)
        if cv_a != cv_b:
            # calculate similarity:
            if (len(pep_a) == len(pep_b)) and (mhc_a == mhc_b):
                # identity = seq_identity(pep_a, pep_b)
                # print identity, '\t', pep_a, pep_b
                if is_similar(pep_a, pep_b, sim_cutoff=0.80):
                    count_sim_peptides = count_sim_peptides + 1
                    if debug:
                        print(line_a.strip())
                        print(line_b.strip())
                        print('Warning: similar peptides between cv-groups', pep_a, pep_b)
                        print('\n')
    return count_sim_peptides

def count_faulty_meas(fname_cv, debug=False):
    # fname_bdata = sys.argv[1]
    print('DEBUG INPUT cross-validation data set =', fname_cv)
    lines = open(fname_cv, 'r').readlines()
    count_faulty_meas = 0
    for i in range(1, len(lines)):
        line = lines[i]
        (pep, cv, mhc, plen, ineq, meas_ic50) = get_peptide(line)
        if is_faulty_meas(ineq, meas_ic50):
            count_faulty_meas = count_faulty_meas + 1

    return count_faulty_meas

def is_faulty_meas(inequality, meas_ic50):
    """
    """
    meas_ic50 = float(meas_ic50)
    meas_faulty = False
    if (inequality == '<') and (meas_ic50 <= 10000.0) and (meas_ic50 > 20.0):
        meas_faulty = True
    return meas_faulty

def test_check_cv_data_sets():
    """
    Make sure there are no similar peptides between cv-partitions for cv_gs.

    To run the test, issue the following command:

    nosetests quick-check-cv-data.py
    """
    fname_cv_rnd = 'bdata.2009.mhci.public.1.cv_rnd.txt'
    fname_cv_sr = 'bdata.2009.mhci.public.1.cv_sr.txt'
    fname_cv_gs = 'bdata.2009.mhci.public.1.cv_gs.txt'
    count_rnd = count_peptides_similar(fname_cv_rnd)
    count_sr = count_peptides_similar(fname_cv_sr)
    count_gs = count_peptides_similar(fname_cv_gs)

    print('count_rnd', count_rnd)
    print('count_sr', count_sr)
    print('count_gs', count_gs)

    assert count_gs == 0
    assert count_sr < count_rnd

def test_faulty_meas():
    """
    Make sure there are no 'faulty' measurements. See 'is_faulty_meas' for definition.

    To run the test, issue the following command:

    nosetests quick-check-cv-data.py
    """
    fname_cv_rnd = 'bdata.2009.mhci.public.1.cv_rnd.txt'
    fname_cv_sr = 'bdata.2009.mhci.public.1.cv_sr.txt'
    fname_cv_gs = 'bdata.2009.mhci.public.1.cv_gs.txt'

    # fname_cv_gs = '../bd2009/bdata.2009.mhci.public.txt'

    count_rnd = count_faulty_meas(fname_cv_rnd)
    count_sr = count_faulty_meas(fname_cv_sr)
    count_gs = count_faulty_meas(fname_cv_gs)

    print('count_rnd', count_rnd)
    print('count_sr', count_sr)
    print('count_gs', count_gs)

    assert count_rnd == 0
    assert count_sr == 0
    assert count_gs == 0
