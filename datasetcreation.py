## This file's purpose is to create a dataset that is usable and similar in style to the standard CodeSearchNet dataset.

"""
Usage:
    datasetcreation.py [options] SAVE_TRAIN SAVE_TEST SAVE_VALID

DESCRIPTION: Creates the desired dataset from the funcom dataset

SAVE_TRAIN is the path to the place where the training dataset is to be saved.
SAVE_VALID is the path to the place where the validation dataset is to be saved.
SAVE_TEST is the path to the place where the testing dataset is to be saved.


Options:
    -h --help                  Show this screen.
    --split STR                The split that the dataset uses for test and validation data. They will be of equal size.
    --debug                    Enable debug routines. [default: False]
"""


import sys
from docopt import docopt
import pandas as pd
import time
import nltk
from dpu_utils.utils import run_and_debug, RichPath

sys.path.insert(1, '/home/marcbo/dataprep/funcomgen/') #path to utils directory in next line
from utils import pkldf2jsonl #utils file is from the CodeSearchNet utils directory on Github

def run(arguments):
    split = int(arguments['--split'])

    pd.set_option('display.max_colwidth', None) #NEEDED otherwise str typecasting will cut off the function at its preset limit
    functioncode = pd.read_json('/itet-stor/marcbo/net_scratch/funcomprep/funcom_processed/functions.json',orient='index',typ='frame')

    filedesc = pd.read_json('/itet-stor/marcbo/net_scratch/funcomprep/funcom_processed/comments.json',orient='index',typ='frame')

    df1test = pd.DataFrame(columns=['code','code_tokens','docstring','docstring_tokens','partition'])
    df1train = pd.DataFrame(columns=['code','code_tokens','docstring','docstring_tokens','partition'])
    df1valid = pd.DataFrame(columns=['code','code_tokens','docstring','docstring_tokens','partition'])

    maindftrain = pd.DataFrame(columns=['code','code_tokens','docstring','docstring_tokens','partition'])
    maindftest = pd.DataFrame(columns=['code','code_tokens','docstring','docstring_tokens','partition'])
    maindfvalid = pd.DataFrame(columns=['code','code_tokens','docstring','docstring_tokens','partition'])

    i = 0
    time1 = time.time()
    
    for i in range(0,functioncode.size):  
        code = str(functioncode.iloc[[i]])
        code = code.split(maxsplit=2)[2]
        codetoken = nltk.wordpunct_tokenize(code)


        if df1test.size == 5000 and i <= split:
            maindftest = pd.concat([maindftest,df1test])
            df1test = pd.DataFrame(columns=['code','code_tokens','docstring','docstring_tokens','partition'])

        elif df1valid.size == 5000 and i > split and i <= 2*split:
            maindfvalid = pd.concat([maindfvalid,df1valid])
            df1valid = pd.DataFrame(columns=['code','code_tokens','docstring','docstring_tokens','partition'])

        elif df1train.size == 5000 and i > 2*split:
            maindftrain = pd.concat([maindftrain,df1train])
            df1train = pd.DataFrame(columns=['code','code_tokens','docstring','docstring_tokens','partition'])
        else:
            pass

        doc = str(filedesc.iloc[[i]])
        doc = doc.split(maxsplit=2)[2]
        # doc = doc.split(" <s> ") 
        # doc = doc[1]
        # doc = doc.split(" </s>")
        # doc = doc[0]
        doctoken = doc.split()

        if (i <= split):
            partition = "test"
            df2 = pd.DataFrame([[code,codetoken,doc,doctoken,partition]],columns=['code','code_tokens','docstring','docstring_tokens','partition'])
            df1test = pd.concat([df1test,df2])

        elif (i > split and i <= 2*split):
            partition = "valid"
            df2 = pd.DataFrame([[code,codetoken,doc,doctoken,partition]],columns=['code','code_tokens','docstring','docstring_tokens','partition'])
            df1valid = pd.concat([df1valid,df2])

        else:
            partition = "train"
            df2 = pd.DataFrame([[code,codetoken,doc,doctoken,partition]],columns=['code','code_tokens','docstring','docstring_tokens','partition'])
            df1train = pd.concat([df1train,df2])
        

        if i % 5000 == 0 and i != 0:
            time2 = time.time()
            timediff = time2 - time1
            print(i)
            print(doc)
            print(code)
            print(round(timediff,2)," seconds for last 5k lines")
            time1 = time2
        
    maindftrain = pd.concat([maindftrain,df1train]) #Fill up the last iteration
    print(maindftest)

    save_folder_train = RichPath.create(arguments['SAVE_TRAIN'])
    save_folder_test = RichPath.create(arguments['SAVE_TEST'])
    save_folder_valid = RichPath.create(arguments['SAVE_VALID'])

    # print(maindftrain)
    # print(maindftest)
    # print(maindfvalid)

    pkldf2jsonl.df_to_jsonl(maindftrain,save_folder_train,i=0)
    pkldf2jsonl.df_to_jsonl(maindftest,save_folder_test,i=0)
    pkldf2jsonl.df_to_jsonl(maindfvalid,save_folder_valid,i=0)

if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
