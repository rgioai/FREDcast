#!/usr/bin/env python3

import subprocess
import os
import sys


def show_help(verbosity=1):
    if verbosity == 1:
        print('Basic usage for scan analysis:\n'
              '. nstc.py <classifier_to_use> <absolute_path_to_scan_directory>\n'
              'To see additional production classifer usage:\n'
              '. nstc.py -h 2\n'
              'To see training usage:\n'
              '. nstc.py --help\n')
    elif verbosity == 2:
        print('Production classifier usage:\n'
              '. nstc.py <classifier_to_use> <absolute_path_to_scan_directory>\n'
              'ex: . nstc.py -1 /mypatient\n'
              'To download project repository:\n'
              '. nstc.py --setup\n'
              'To install or update dependencies:\n'
              '. nstc.py <classifier> -r\n'
              'To make a competition submission:\n'
              '. nstc.py <classifier> -c <absolute_path_to_competition_tld>\n'
              'where competition_tld mirrors /testing structure from deployment_file_structure\n')
    elif verbosity == 3:
        print('Retraining usage:\n'
              'WARNING: These operations require GPU acceleration,\n'
              'approx ~1TB of storage, and days of operation for many functions.\n\n'
              'To download repository and data:\n'
              '. nstc.py --training --setup\n'
              'To install or update dependencies:\n'
              '. nstc.py --training -r\n'
              'To retrain a classifier:'
              '. nstc.py --training --retrain <classifier>\n'
              'ex: . nstc.py --training --retrain -1\n')
    else:
        raise ValueError


def require_dir(usr_path):
    if len(usr_path.split('/')) > 2:
        pass
        # TODO Upgrade to handle multiple directory creation
        # i.e. require_dir('/storage/NSTC') first runs require_dir('/storage') then require_dir('/storage/NSTC')
    if not os.path.exists(usr_path):
        os.mkdir(usr_path)


def download_code():
    require_dir('/storage')
    require_dir('/storage/nstc')
    os.chdir('/storage/nstc')
    if not os.path.exists('/storage/nstc/DataScienceBowl17'):
        subprocess.run(['git', 'clone', 'git@github.com:gioGats/DataScienceBowl17.git'])
    else:
        os.chdir('/storage/nstc/DataScienceBowl17')
        subprocess.run(['git', 'stash'])
        subprocess.run(['git', 'pull'])


def project_requirements(clf):
    if clf == '1':
        os.chdir('/storage/nstc/DataScienceBowl17/production_1')
    elif clf == '2':
        os.chdir('/storage/nstc/DataScienceBowl17/production_2')
    elif clf == 'x':
        os.chdir('/storage/nstc/DataScienceBowl17/experimentation')
    else:
        raise ValueError
    subprocess.run(['pip3', 'install', '--upgrade', '-r', 'requirements.txt'])


def download_training_data():
    require_dir('/storage/nstc')
    os.chdir('/storage/nstc')
    raise NotImplementedError
    # Will need to identify some large file hosting capability
    # subprocess.run(['scp', 'ftp@some.host:/storage/nstc/compressed_data.tar.lrz', 'raw_data.tar.lrz'])
    # subprocess.run(['lrztar', '-d', 'raw_data.tar.lrz'])
    # subprocess.run(['rm', 'raw_data.tar.lrz'])


def structure_training_data():
    raise NotImplementedError
    # os.chdir('/storage/nstc/raw_data')
    # subprocess.run(['cp', '/storage/nstc/DataScienceBowl17/misc/sams_script.sh', '/storage/nstc/raw_data/'])
    # subprocess.run(['.', 'sams_script.sh'])
    # subprocess.run(['rm', 'sams_script.sh'])
    # subprocess.run(['rm', '*.tar.gz'])


def retrain_production_1():
    raise NotImplementedError


def retrain_production_2():
    raise NotImplementedError


def classify(clf, patient_dir):
    if clf == '1':
        os.chdir('/storage/nstc/DataScienceBowl17/production_1')
    elif clf == '2':
        os.chdir('/storage/nstc/DataScienceBowl17/production_2')
    # TODO Load the model and classify
    else:
        raise ValueError('Invalid classifier')


def make_submission(clf, competition_tld):
    i = 0
    while os.path.exists('/storage/nstc/testing/submission_clf%s_%d.csv' % (clf, i)):
        i += 1
    with open('', 'w') as f:
        f.write('patient_id,cancer\n')
        for patient_dir in os.listdir(competition_tld + '/test_images'):
            prediction = classify(clf, '%s/test_images/%s' % (competition_tld, patient_dir))
            f.write('%s,%.4f\n' % (patient_dir, prediction))


if __name__ == '__main__':
    if '-h' in sys.argv or len(sys.argv) < 2:
        try:
            v = int(sys.argv[sys.argv.index('-h') + 1])
        except ValueError:
            v = 1
        show_help(verbosity=v)
    elif '--help' in sys.argv:
        show_help(verbosity=3)
    elif '--training' in sys.argv:
        print('WARNING: You have selected training-level commands.\n'
              'These operations require GPU acceleration, approx ~1TB of storage, '
              'and days of operation for many functions.\n')
        if input('Continue? [y/n] ').lower() == 'y':
            if '--setup' in sys.argv:
                print('WARNING: Initiating a ~120GB download.\n'
                      'Depending on connection speed, you need to either take a coffee break or a summer vacation.\n'
                      'Decompression and structuring will ultimately require ~500GB of disk space\n'
                      'YOU HAVE BEEN WARNED')
                if input('Continue? [y/n]').lower() == 'y':
                    download_code()
                    download_training_data()
                    structure_training_data()
                else:
                    sys.exit()
            if '--setup' in sys.argv or '-r' in sys.argv:
                project_requirements('x')
            if '--retrain' in sys.argv:
                print('WARNING: Initiating a classifer retraining.\n'
                      'This will overwrite the pre-trained model available via the git repository.\n'
                      'It will not alter model design or training parameters.\n'
                      'Such alterations must be done via source code modifications.\n'
                      'Depending on your hardware, retraining may require either a coffee break or a summer vacation.\n'
                      'YOU HAVE BEEN WARNED.')
                if input('Continue? [y/n] ').lower() == 'y':
                    if '-1' in sys.argv:
                        retrain_production_1()
                    elif '-2' in sys.argv:
                        retrain_production_2()
                    else:
                        raise ValueError('Classifier not specified')
                else:
                    sys.exit()
        else:
            sys.exit()

    elif '--setup' in sys.argv:
        download_code()

    else:
        if '-1' in sys.argv:
            usr_clf = '1'
        elif '-2' in sys.argv:
            usr_clf = '2'
        else:
            raise ValueError('Classifier not specified')

        if '-r' in sys.argv:
            project_requirements(usr_clf)
        elif '-c' in sys.argv:
            usr_comp_tld = sys.argv[-1]
            make_submission(usr_clf, usr_comp_tld)
        else:
            usr_patient_dir = sys.argv[-1]
            usr_patient_id = usr_patient_dir.split('/')[-1]
            usr_patient_prob = classify(usr_clf, usr_patient_dir)
            print('Patient %s lung cancer probability is %.4f' % (usr_patient_id, usr_patient_prob))
