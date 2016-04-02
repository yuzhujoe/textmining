import json
import re
import Queue
import string
import time
import math
from scipy import sparse
import numpy
import numpy.matlib

train_filename = "./resources/yelp_reviews_train.json"
dev_filename = "./resources/yelp_reviews_dev.json"
dev_filename2 = "./resources/yelp_reviews_train2.json"

train_ctf_token_clean_filename = "train_ctf_review.clean"
dev_ctf_token_clean_filename = "dev_ctf_review.clean"
dev_ctf_token_clean_filename2 = "train_ctf_review2.clean"

train_output_filename = "./resources/yelp_reviews_train.json"
dev_predict_output_filename = "dev_predict.out"

stop_word_file = "./resources/stopword.list"
ctf_filename = "top2000CTF"
df_filename = "top2000DF"
re_d = re.compile('\d')
table = string.maketrans("","")
punc = string.punctuation
num_feature = 2000
train_y_filename = "star"
learning_rate = 0.001
num_cat = 5
lmd = 0.01
stop_criteria = 1e-7
stop_criteria_w_gradient = 1e-2
batch_size = 100
eval_percent = 0.05
max_iter = 50


def load_stop_word(stop_word_file):
    l = []
    with open(stop_word_file) as f:
        for lines in f:
            sw = lines.strip()
            l.append(sw)
    stop_word_set = set(l)
    return stop_word_set

def tokenization(input_str,stop_word_set):
    tokens = re.split('\\s',input_str.strip())
    reslist = []
    for token in tokens:
        token  = token.lower()
        if re_d.search(token) == None:
            # TODO HYPHEN
            # token = re.sub("[.!?',:;()&-\\/]",'',token)
            token = token.translate(table, punc)
            if token != '' and (token not in stop_word_set):
                    reslist.append(token)
    return reslist


def process_test_dev_JSON(filename,stop_word_set, num_of_data = None):
    all_doc_token = []
    linen = 0
    with open(filename) as f:
        for lines in f:
            js = json.loads(lines)
            text = js["text"]
            try:
                review_text =  str(text)
            except UnicodeEncodeError:
                text = re.sub(r"[^\x00-\x7F]+",'', text)
                review_text = str(text)
            # if stars not in hm_star:
            #     hm_star[stars] = 1
            # else:
            #     hm_star[stars] += 1
            # st = time.time()
            tokenlist = tokenization(review_text,stop_word_set)
            # print "time: ",time.time() - st
            # TODO : PUT TOKENLIST INTO HASHMAP FOR STATUSTICS
            doc_hm = {}

            for t in tokenlist:
                if t not in doc_hm:
                    doc_hm[t] = 1
                else:
                    doc_hm[t] += 1

            all_doc_token.append(doc_hm)

            if num_of_data != None:
                linen += 1
                if linen >= num_of_data:
                    break


    return all_doc_token

def process_train_JSON(train_filename, num_of_data = None):
    stop_word_set = load_stop_word(stop_word_file)
    hm = {}
    all_doc_token = []
    linen = 0

    with open(train_filename) as f:
        for lines in f:
            js = json.loads(lines)

            text = js["text"]
            try:
                review_text =  str(text)
            except UnicodeEncodeError:
                text = re.sub(r"[^\x00-\x7F]+",'', text)
                review_text = str(text)

            stars = js["stars"]

            # if stars not in hm_star:
            #     hm_star[stars] = 1
            # else:
            #     hm_star[stars] += 1
            # st = time.time()
            tokenlist = tokenization(review_text,stop_word_set)
            # print "time: ",time.time() - st
            # TODO : PUT TOKENLIST INTO HASHMAP FOR STATUSTICS
            doc_hm = {}

            for t in tokenlist:
                if t not in hm:
                    hm[t] = 1
                else:
                    hm[t] += 1
                if t not in doc_hm:
                    doc_hm[t] = 1
                else:
                    doc_hm[t] += 1

            all_doc_token.append(doc_hm)

            if num_of_data != None:
                linen += 1
                if linen >= num_of_data:
                    break


    return [hm,all_doc_token]

def process_train_star(train_filename):
    starlist = []
    with open(train_filename) as f:
        for lines in f:
            js = json.loads(lines)
            stars = js["stars"]
            starlist.append(stars)

    return starlist

def save_train_star():
    sl = process_train_star(train_filename)
    with open("star","w") as f:
        for s in sl:
            f.write(str(s)+"\n")

def load_train_star():
    res = []
    with open(train_y_filename,"r") as f:
        for line in f:
            res.append(int(line.strip()))

    return res

def process_train_JSON_DF(train_filename):
    stop_word_set = load_stop_word(stop_word_file)
    hm_df = {}

    linen = 0
    with open(train_filename) as f:
        for lines in f:
            js = json.loads(lines)

            text = js["text"]
            try:
                review_text =  js["text"].encode('utf-8')
            except UnicodeEncodeError:
                text = re.sub(r"[^\x00-\x7F]+",'', text)
                review_text = str(text)

            # st = time.time()
            tokenlist = tokenization(review_text,stop_word_set)
            # print "time: ",time.time() - st
            # TODO : PUT TOKENLIST INTO HASHMAP FOR STATISTICS
            tknset =  set(tokenlist)
            for t in tknset:
                if t not in hm_df:
                    hm_df[t] = 1
                else:
                    hm_df[t] += 1

            linen += 1
            if linen > 10:
                break

    return hm_df

# feature dictionary is built using champion list
def build_dic(top_num,hm):
    q = Queue.PriorityQueue()
    for t in hm:
        if q.qsize() < top_num:
            q.put((hm[t],t))
        else:
            (count,token) = q.queue[0]
            if hm[t] > count:
                q.get()
                q.put((hm[t],t))
    return q



def write_dic_to_file(q,top_num,filename):
    qlist = q.queue
    with open(filename,"w") as f:
        for i in xrange(top_num):
            (count, token) = qlist[i]
            f.write(str(count)+"\t"+token+"\n")

# generate dictionary from top 2000 CTF word
def read_dic_from_file(filename):
    dic = {}
    idx = 0
    with open(filename,'r') as f:
        for line in f:
            count, word = line.strip().split('\t')
            dic[word] = idx
            idx += 1

    return dic

def build_feature_vector(dic,all_doc_token):
    l = len(all_doc_token)
    print "l: ",l

    rowlist = []
    collist = []
    vallist = []

    for i in xrange(l):
        d = all_doc_token[i]
        for t in d:
            rowlist.append(i)
            collist.append(t)
            vallist.append(d[t])

    print "finish load"
    s = sparse.csr_matrix((vallist,(rowlist,collist)),shape = (l,num_feature))
    print "finish construct"

    return s

# compute RMLR
# W: C*2000
def gradient_one_data(W,c,xi,yi):
    return (yi[0,c] - predict_prob(W,c,xi))*xi - lmd*W[c,:]


def gradient_mini_batch_data(W,c,X,Y):
    numrow,numcol = X.shape

    # st = time.time()
    tmp = numpy.zeros((1,numrow))

    st = time.time()
    for i in xrange(numrow):
        tmp[0,i] = (Y[i,c] - predict_prob(W,c,X[i,:]))
    g  = tmp * X

    print "compute gradient: ", time.time() - st

    # for i in xrange(numrow):
    #     # st = time.time()
    #     tmp = (Y[i,c] - predict_prob(W,c,X[i,:]))*X[i,:]
    #     # print "compute gradient: ", time.time() - st
    #     st = time.time()
    #     g += tmp
    #     print "compute gradient: ", time.time() - st
    return g - lmd*W[c,:]



def gradient_mini_batch_all_cat(W,X,Y):
    # st = time.time()
    print "W: ",numpy.linalg.norm(W)
    # print "X: ",numpy.linalg.norm(X)

    xw = X* W.transpose()
    xw = numpy.exp(xw)
    xwsum = numpy.sum(xw,1,keepdims=True)

    xws = xw/xwsum

    # print "xws: ",xws

    g = (Y - xws).transpose() * X - lmd * W


    return g

def predict_prob(W,c,xi):
    sum = 0
    x = xi.transpose()

    for i in range(len(W)):
        sum += math.exp(W[i,:]*x)

    return math.exp(W[c,:]*x)/sum

def hard_predict(W,X):
    xw = X* W.transpose()

    xw = numpy.exp(xw)

    xwsum = numpy.sum(xw,1,keepdims=True)

    xws = xw/xwsum

    xwsmax= numpy.argmax(xws,1)


    xwsmax = numpy.ones(shape = xwsmax.shape) + xwsmax
    return xwsmax.tolist()

def soft_predict(W,X):

    # soft_prob_cat = 0
    # for i in xrange(num_cat):
    #     pp = predict_prob(W,i,xi)
    #     soft_prob_cat += (i+1)*pp

    numrow,numcol = X.shape
    xw = X* W.transpose()
    xw = numpy.exp(xw)
    xwsum = numpy.sum(xw,1,keepdims=True)

    xws = xw/xwsum

    print
    nmr = numpy.matlib.repmat(numpy.arange(1.0,6.0),numrow,1)

    xwsps = numpy.multiply(xws,nmr)

    xwspsum = numpy.sum(xwsps,axis=1)

    return xwspsum.tolist()

def write_predict_2_file(W,X,filename):

    with open(filename,"w") as f:
        numrow, numcol = X.shape
        hard = int(hard_predict(W,X))
        soft = soft_predict(W,X)

        for i in xrange(numrow):
            f.write(str(hard[i]) + " "+ str(soft[i]) + "\n")



def log_likelihood(W,X,Y):

    xw = X* W.transpose()
    xw = numpy.exp(xw)
    xwsum = numpy.sum(xw,1,keepdims=True)

    xws = xw/xwsum

    xwsp = numpy.multiply(xws,Y)

    xwspsum = numpy.sum(xwsp,1,keepdims=True)

    xwspslog = numpy.log(xwspsum)

    xwspslogs = numpy.sum(xwspslog)
    nm = numpy.linalg.norm(W)
    return xwspslogs - lmd/2 * nm * nm


    # g = (Y - xws).transpose() * X - lmd * W

    # numrow, numcol = X.shape
    # sum = 0
    # for i in xrange(numrow):
    #     c = Y[i] - 1
    #     sum += math.log(predict_prob(W,c,X[i,:]))
    #
    # sum -= (lmd/2 * numpy.linalg.norm(W))




def star_to_vector(star):
    numrow = len(star)
    Y = numpy.zeros((numrow,num_cat))
    for i in xrange(numrow):
        Y[i,star[i]-1] = 1

    return Y


def sgd_one_data(alpha,W,s,Y):
    numrow,numcol = s.shape
    wrow,wcol = W.shape
    break_outer_loop = False

    while True:
        for j in xrange(numrow):
            Wnew = numpy.zeros((wrow,wcol))
            for i in xrange(num_cat):
                wg = gradient_one_data(W,i,s[j,:],Y)
                Wnew[i,:] = W[i,:] +  alpha * wg

            if check_stop_criteria_w(Wnew,W):
                break_outer_loop = True
                break
            W = Wnew.copy()

        if break_outer_loop:
            break
    return W

def sgd_mini_batch(alpha,batch_size,W,s,Y):

    numrow,numcol = s.shape
    wrow,wcol = W.shape
    break_outer_loop = False

    validate_idx = partition_data_2_validateset(numrow)

    Yval = [Y[i] for i in validate_idx]

    Yval_vec = star_to_vector(Yval)

    batch_idx = partition_data_2_batch(batch_size,numrow)

    Wold = W

    Y_vec = star_to_vector(Y)
    count = 0
    llnew = None
    llold = None

    iter = 0
    while iter < max_iter:
        for j in xrange(len(batch_idx)):
            data_idx = batch_idx[j]
            # Wnew = numpy.zeros((wrow,wcol))

            st = time.time()

            wg = gradient_mini_batch_all_cat(Wold,s[data_idx,:],Y_vec[data_idx,:])
            print "r1: ",time.time() - st
            Wnew = Wold + alpha * wg
            print "gradient: ",numpy.linalg.norm(wg)

            # st = time.time()
            # if check_stop_criteria_w(Wnew,W):
            #     break_outer_loop = True
            #     break
            # if check_stop_criteria_gradient(wg):
            #     break_outer_loop = True
            #     break

            # print "check time: ", time.time()- st

            st = time.time()
            Wold = Wnew.copy()
            print "r2: ",time.time() - st

            st = time.time()
            llnew = log_likelihood(Wold,s[validate_idx,:],Yval_vec)
            print "r3: ",time.time() - st

            print "lnew: ",llnew


            print "lold: ",llold
            if check_stop_criteria_loglikelihood(llold,llnew):
                break_outer_loop = True
                break
            llold = llnew

            print "round : %s" %count, " log likelihood: %s" %llold
            # print "round : %s" %count
            count += 1
        if break_outer_loop:
            break
        iter += 1
    return Wold



def partition_data_2_batch(batch_size,numrow):
    res = []
    tmp = []
    size = 0
    startidx = int(eval_percent * numrow)
    for i in xrange(startidx,numrow):
        tmp.append(i)
        size += 1
        if size == batch_size or i == numrow - 1:
            res.append(tmp)
            tmp = []
            size = 0

    return res

def partition_data_2_validateset(numrow):
    res = []
    # todo change to other selection method
    len = int(eval_percent * numrow)
    for i in xrange(len):
        res.append(i)

    return res


def check_stop_criteria_w(Wnew,Wold):
    norm_wnew = numpy.linalg.norm(Wnew)
    norm_wold = numpy.linalg.norm(Wold)

    if abs(norm_wnew - norm_wold) < stop_criteria:
        return True
    else:
        return False

def check_stop_criteria_gradient(wg):
    norm_wg = numpy.linalg.norm(wg)

    if abs(norm_wg) < stop_criteria_w_gradient:
        return True
    else:
        return False

def check_stop_criteria_loglikelihood(llold,llnew):
    if llold == None:
        return False
    elif abs((llnew - llold)/llold) < stop_criteria:
        print "ratio: ", abs((llnew - llold)/llold)
        return True
    else:
        return False

def write_doc_token_2_file(filename,doc_token,dic):
    with open(filename,"w") as f:
        for i in xrange(len(doc_token)):
            for token in doc_token[i]:
                cnt = doc_token[i][token]
                if token in dic:
                    f.write(str(dic[token]) + ":"+ str(cnt)+" ")
            f.write("\n")

def read_doc_token_from_file(filename,dic):
    all_doc_token = []
    with open(filename,"r") as f:
        for line in f:
            tokens = line.strip().split(" ")
            hm = {}
            for t in tokens:
                try:
                    idx,cnt = t.split(":")
                    hm[int(idx)] = int(cnt)
                except ValueError:
                    pass
            all_doc_token.append(hm)

    return all_doc_token

if __name__ == "__main__":

    # Y = load_train_star()[0:1000]
    Y = load_train_star()

    # st = time.time()

    # hm, all_doc_token = process_train_JSON(train_filename)
    # print "finish train: ",time.time() - st
    dic = read_dic_from_file(ctf_filename)

    all_doc_token = read_doc_token_from_file(train_ctf_token_clean_filename,dic)

    # write_doc_token_2_file(train_ctf_token_clean_filename,all_doc_token,dic)

    X = build_feature_vector(dic,all_doc_token)

    W_init = numpy.ones((num_cat,num_feature))*0.2

    W_final = sgd_mini_batch(learning_rate,batch_size,W_init,X,Y)

    stop_word_set = load_stop_word(stop_word_file)
    # # dev_doc_token = process_test_dev_JSON(dev_filename,stop_word_set)

    dev_doc_token = read_doc_token_from_file(dev_ctf_token_clean_filename,dic)
    print dev_doc_token
    # write_doc_token_2_file(dev_ctf_token_clean_filename2,dev_doc_token,dic)
    X_dev = build_feature_vector(dic,dev_doc_token)


    write_predict_2_file(W_final,X_dev,dev_predict_output_filename)


    # load_stop_word(stop_word_file)
    # st = time.time()
    # hm,all = process_train_JSON_DF(train_filename)
    # print "time: ",time.time() - st
    #
    # q = Queue.PriorityQueue()
    # for t in hm:
    #     if q.qsize() < top_limit:
    #         q.put((hm[t],t))
    #     else:
    #         (count,token) = q.queue[0]
    #         if hm[t] > count:
    #             q.get()
    #             q.put((hm[t],t))
    #
    # with open(df_filename,"w") as f:
    #     for i in xrange(top_limit):
    #         (count, token) = q.get()
    #         f.write(str(count)+"\t"+token+"\n")

