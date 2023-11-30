import os
from tqdm import tqdm
import random
import keyword
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import time
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

num_clusters = 10 #number of clusters in the index

def Sort(sub_li):finallyoutput = []; sort_sub_li = sorted(sub_li.items(), key=lambda x: x[1], reverse=True);return sort_sub_li #to sort dictionaries

datasets = ["ecommerce.txt","chainstore.txt","fruithut_utility.txt","liquor_15.txt"] #names of datasets
folders = ['Ecommerce','Fruithut'] #names of folders - on which experiments were run

def getkws(kui_entry,kword_dict): #get keywords
	item_set = kui_entry[0];combined_tuple = tuple(max(kword_dict[item][i] for item in item_set) for i in range(uni_k))
	return combined_tuple

def getprice(itemset,prices): #get price of itemset
	p = 0
	for each in itemset:
		p = p + prices[each]
	return p

def commonclusterfinder(list_of_lists,input_list): #given a list of sets of keywords, and a set of queried keywords, finds the top-lists of sets of keywords with matching keywords
	max_common = max(map(lambda x: len(set(input_list) & set(x)), list_of_lists))
	similar_lists = [lst for lst in list_of_lists if len(set(input_list) & set(lst)) == max_common]
	return similar_lists

def huifinder(list_of_lists,input_list): #gets the matching score between the top-list of lists and queried keyword list
	max_common = max(map(lambda x: len(set(input_list) & set(x)), list_of_lists))
	similar_lists = [lst for lst in list_of_lists if len(set(input_list) & set(lst)) == max_common]

	return max_common

def makesets(): #create testing training sets
	try:os.mkdir("Ecommerce");os.mkdir("Fruithut");os.mkdir("Liquor");os.mkdir("Chainstore")
	except:print("Folder exists")
	d1 = open("fruithut_utility.txt","r");d2 = open("liquor_15.txt","r");d3=open("chainstore.txt","r");d4=open("ecommerce.txt","r")
	d1trans = d1.readlines(); d1len = len(d1trans); d1train = int(d1len * 0.8);d2trans = d2.readlines(); d2len = len(d2trans); d2train = int(d2len * 0.8);d3trans = d3.readlines(); d3len = len(d3trans); d3train = int(d3len * 0.8);d4trans = d4.readlines(); d4len = len(d4trans); d4train = int(d4len * 0.8)
	traind1 = open("Fruithut/train.txt","w"); testd1 = open("Fruithut/test.txt","w");traind2 = open("Liquor/train.txt","w"); testd2 = open("Liquor/test.txt","w");traind3 = open("Chainstore/train.txt","w"); testd3 = open("Chainstore/test.txt","w");traind4 = open("Ecommerce/train.txt","w"); testd4 = open("Ecommerce/test.txt","w")
	traind1.writelines("%s" % item for item in d1trans[:d1train]);testd1.writelines("%s" % item for item in d1trans[d1train:]);traind2.writelines("%s" % item for item in d2trans[:d2train]);testd2.writelines("%s" % item for item in d2trans[d2train:]);traind3.writelines("%s" % item for item in d3trans[:d3train]);testd3.writelines("%s" % item for item in d3trans[d3train:]);traind4.writelines("%s" % item for item in d4trans[:d4train]);testd4.writelines("%s" % item for item in d4trans[d4train:])
makesets()

def generateprice(link): #get price of items from dataset
	dictionaryofprice = {};file = open(link,"r"); retaildata = file.readlines()
	for each in retaildata:
		each = each.strip();each = each.split(":");temp = each[0];temp = temp.split(" ");temp1 = each[-1];temp1 = temp1.split(" ");temp = [int(item) for item in temp]
		for item in temp:
			try:
				price = dictionaryofprice[item];ind = temp.index(item);newprice = float(temp1[ind])
				if newprice < price:dictionaryofprice[item] = newprice
			except:
				ind = temp.index(item);dictionaryofprice[item] = float(temp1[ind])
	return dictionaryofprice

def generatefreq(link): #get frequency of sale from dataset
	dictionary = {}
	file = open(link,"r");retaildata = file.readlines()
	for each in retaildata:
		each = each.strip();each = each.split(":");each=each[0];each = each.split(" ")
		each = [int(item) for item in each]
		for item in each:
			try:frequency = dictionary[item];frequency += 1;dictionary[item] = frequency
			except:dictionary[item] = 1
	return (dictionary)

def commonclusterfinder(list_of_lists,input_list):  #given a list of sets of keywords, and a set of queried keywords, finds the top-lists of sets of keywords with matching keywords
	max_common = max(map(lambda x: len(set(input_list) & set(x)), list_of_lists))
	similar_lists = [lst for lst in list_of_lists if len(set(input_list) & set(lst)) == max_common]
	return similar_lists

#to create itemsets, we follow the following approach:

#Itemsets are mined in a level-wise manner. At the onset, we scan the transactional dataset D. 
#The first level contains the top-lambda high-utility items. To build level 2, we scan the transactional
#dataset D. Note that each item in D has a unique identifier, designated as IID, which is an integer. If item
# i occurs in a transaction t, we use the remaining items (of t) occurring at level 1 such that they have higher
# IID than that of i. For example, while iterating through the items in level 1, consider item 5 as the input item
# for creating itemsets for level 2, and consider a sample transaction {3,5,6,7}. Assume that items 3,5,6 and 7
# occur at level 1. Now we consider the remaining items with a higher IID than 5 i.e., {6,7} and use them to create itemsets {5,6}, {5,7}.

#Intuitively, {3,5} should also qualify for level 2; here, {3,5} will be considered when 3 is the input item for creating itemsets of size 2.
# If {5,6} and {5,7} already exist in the itemset linked list of level 2, we simply increment their frequencies by 1, thereby saving
# time in computing itemset frequencies. Observe how by following the above approach, we avoid generating duplicate itemsets
# e.g., {A,B} and {B,A}. Once itemsets have been generated for level 2 as discussed above, we sort the itemsets in
# descending order of utility. Then, we progressively add itemsets at level 2 of the index, if and only if they belong to the top-lambda sorted itemsets.

#For creating higher levels, say level n, we essentially follow the same process, while considering itemsets of level (n-1) and items at level 1.
# We do so in the following manner. For items in transactions that have lower IIDs than the highest item IID in the input itemset, we create a temporary itemset TI
# with the remaining items that excludes the item with the highest IID, and check if TI occurs at the previous level of UPI. For example, assuming that items 3,5,6 and 7
# occur at level 1, for input itemset {6,7} and transaction {3,5,6,7}, we have two TIs, namely {5,6} and {3,6}, as 7 is the item with the highest IID.
# We proceed to check if {5,6} and {3,6} occur at level 2. If not, then we proceed to create the new itemsets, along with the item with the highest IID,
# for the next level, i.e., {5,6,7} and {3,6,7}. We increment their frequencies by 1, thereby saving time in computing itemset frequencies. However,
# if it does occur at the previous level, then we discard the corresponding itemsets, as they would automatically become a part of this level during
# future iterations, i.e., when {5,6} and {3,6} are input itemsets. For items in transactions that have higher IIDs than the highest item IID in the input itemset,
# we adhere to the same process as described earlier.


def make_itemsets(level1,leveln_1,train,prices,frequencies,num_itemsets_at_each_level): #make itemsets

	level1items = [item[0][0] for item in level1];input_itemsets = [item[0] for item in leveln_1];output = {}
	for itemset in tqdm(input_itemsets):
		for trans in train:
			if set(itemset).issubset(trans):
				items = set(trans) - set(itemset)
				for item in items:
					if (item > max(itemset)) and (item in level1items):
						newitemset = tuple(itemset + [item])
						try:
							freq = output[newitemset];freq = freq + 1;output[newitemset] = freq
						except:
							output[newitemset] = 1
					elif (item in level1items):
						newitemset = sorted([item]+itemset)
						if newitemset[:-1] not in input_itemsets:
							newitemset = tuple(newitemset)
							try:
								freq = output[newitemset];freq += 1;output[newitemset] = freq
							except:
								output[newitemset] = 1
	output = {key: value * getprice(key,prices) for key, value in output.items()}
	output = sorted(output.items(), key=lambda x: x[0]);output = output[:num_itemsets_at_each_level]; output = [[list(sublist), number] for (sublist, number) in output];output = [item + [getprice(item[0],prices)] for item in output]
	return output

for folder in folders:

	print(folder)

	num_itemsets_at_each_level = 5000;utility_threshold = 0; support_threshold = 0;namedict = {};keyword_sets = []
	train = open(str(folder)+"/train.txt","r");	test = open(str(folder)+"/test.txt","r");train = train.readlines();test = test.readlines()
	train = [item.strip() for item in train];train = [item.split(':') for item in train];train=[item[0] for item in train];train = [item.split(' ') for item in train]
	test = [item.strip() for item in test];test = [item.split(':') for item in test];test=[item[0] for item in test];test = [item.split(' ') for item in test]
	train = [[int(item) for item in inner_list] for inner_list in train];test = [[int(item) for item in inner_list] for inner_list in test]
	prices = generateprice(str(folder)+"/train.txt")
	frequencies = generatefreq(str(folder)+"/train.txt")
	utility = {item: prices[item] * frequencies[item] for item in prices}
	train = [[num for num in inner_list if num in prices] for inner_list in train];test = [[num for num in inner_list if num in prices] for inner_list in test]

	kui_l1 = Sort(utility);kui_l1 = kui_l1[:num_itemsets_at_each_level];kui_l1 = [list(t) for t in kui_l1];kui_l1 = [[[item[0]],item[1]] for item in kui_l1];kui_l1 = [item + [getprice(item[0],prices)] for item in kui_l1]
	level2 = make_itemsets(kui_l1,kui_l1,train,prices,frequencies,num_itemsets_at_each_level)
	level3 = make_itemsets(kui_l1,level2,train,prices,frequencies,num_itemsets_at_each_level)
	level4 = make_itemsets(kui_l1,level3,train,prices,frequencies,num_itemsets_at_each_level)
	level5 = make_itemsets(kui_l1,level4,train,prices,frequencies,num_itemsets_at_each_level)
	level6 = make_itemsets(kui_l1,level5,train,prices,frequencies,num_itemsets_at_each_level)
	total = kui_l1+level2+level3+level4+level5+level6
	total = sorted(total, key=lambda x: x[1], reverse=True)
	util_list = [item[1] for item in total]

	#creation of huis completed above

	if folder== "Ecommerce":
		names = open("ecommerce_item_names.txt","r");lines = names.readlines()
		for line in lines:line = line.strip();line = line.split("=");item = int(line[-2]);name = line[-1];namedict[item] = name
	if folder== "Fruithut":
		names = open("fruithut_names.txt","r");lines = names.readlines()
		for line in lines:line = line.strip();line = line.split("=");item = int(line[-2]);name = line[-1];namedict[item] = name

	python_keywords = keyword.kwlist
	for item in prices:
		if item not in namedict:
			random_keyword = random.choice(python_keywords)
			namedict[item] = random_keyword

	huis = [item[0] for item in total]
	for hui in huis:
		keyword_set = []
		for item in hui:keyword_set.append(namedict[item])
		keyword_sets.append(keyword_set)

	normutility_dict = {}
	for itemset in huis:
		normutility_dict[tuple(itemset)] = (getprice(itemset,prices)/max(util_list))

	#we cluster itemsets below

	itemset_texts = [' '.join(items) for items in keyword_sets]
	vectorizer = CountVectorizer()
	itemset_vectors = vectorizer.fit_transform(itemset_texts)
	kmeans = KMeans(n_clusters=num_clusters)
	kmeans.fit(itemset_vectors)
	clusters = kmeans.labels_

	#we create the upi index below

	all_clusters = []
	upi = {}
	for cluster_id in range(num_clusters):
		cluster_items = [keyword_sets[i] for i in range(len(keyword_sets)) if clusters[i] == cluster_id]
		cluster_items = [item for sublist in cluster_items for item in sublist]
		cluster_items = [word for keyword in cluster_items for word in keyword.split()]
		cluster_items = tuple(set(cluster_items))
		all_clusters.append(cluster_items)
		upi[tuple(cluster_items)]= tuple()

	for name in namedict:
		kw = namedict[name]
		kw = tuple(kw.split(" "))
		namedict[name] = kw

	huikwddict = {}
	global_upi = []

	for hui in huis:
		kws = []
		for item in hui:
			kws += namedict[item]
		kws = list(set(kws))
		global_upi.append([hui,getprice(hui,prices),kws])
		huikwddict[tuple(hui)] = tuple(kws)
		cmon_cluster = commonclusterfinder(all_clusters,kws)
		for cmon_cluste in cmon_cluster:
			res = list(upi[tuple(cmon_cluste)])
			res += [[hui,getprice(hui,prices),kws]]
			res = tuple(res)
			upi[tuple(cmon_cluste)] = res

	numkeys = [15,12,9,6,3] #number of keywords
	w1s = [0]#,0.2,0.4,0.6,0.8,1] #we have assigned full priority to relevance score as input already contains HUIs (w1 = 0)

	for run in range(1): #however many tests you want to run, change 1 accordingly
		print("Run",str(run))
		for numkey in numkeys:
			if numkey == 9:tis = [15,12,9,6,3] #how many HUIs you want
			else:tis = [9]
			for ti in tis:
				for w1 in w1s:
					w2 = 1-w1 #w1 and w2 weights for relevance score and utiltiy
					result_our_prec = 0
					result_ref_prec = 0
					result_our_tu = 0
					result_ref_tu = 0
				
					for each in tqdm(test):

						try:
							itemsets = [];kw = [];prec_dict={};normutility_dict = {};rel_dict = {};ref_kw = [];ref_prec=0;global_prec=[]
							kw += [namedict[item] for item in each if item in prices]
							kw = [item for sublist in kw for item in sublist]
							try:kw = random.choices(kw, k=numkey)
							except:pass
							cluster = commonclusterfinder(all_clusters,kw) #we have simply found the most matching keyword cluster as confidence scores are 1
							itemsets += (upi[tuple(clus)] for clus in cluster);itemsets = itemsets[0]
							for itemset in itemsets:
								prec_dict[tuple(itemset[0])] = len(set(itemset[2])&set(kw))/len(kw)
								# can use the formular below to compute hybrid scores, but we have simply used matching scores (relevance scores) as w1 is zero
								#rel_dict[tuple(itemset[0])] = (w1*(normutility_dict[tuple(itemset[0])])) + (w2*prec_dict[tuple(itemset[0])])
								rel_dict[tuple(itemset[0])] = prec_dict[tuple(itemset[0])]
							for itemset in global_upi:
								global_prec.append(len(set(itemset[2])&set(kw))/len(kw))
							our_scheme = Sort(rel_dict);our_scheme = our_scheme[:ti]
							precision = [prec_dict[t[0]] for t in our_scheme]; precision = sum(precision)/ti
							utility = [getprice(t[0],prices)*prec_dict[t[0]] for t in our_scheme]; utility=sum(utility)
							result_our_prec += precision/max(global_prec)
							result_our_tu += utility
							ref_scheme = random.choices(huis, k=ti)
							ref_kw = [namedict[item] for sublist in ref_scheme for item in sublist]
							for ref_kww in ref_kw:
								ref_prec += len(set(ref_kww)&set(kw))/len(kw)
							ref_prec = ref_prec/ti
							if ref_prec == 0:ref_util = 0.1 * sum([getprice(hui,prices) for hui in ref_scheme])
							else:ref_util = ref_prec * sum([getprice(hui,prices) for hui in ref_scheme])
							result_ref_tu += ref_util
							result_ref_prec += ref_prec/max(global_prec)
						except:pass
					print(numkey,ti,w1)
					print(result_our_prec/len(test),result_ref_prec/len(test),result_our_tu,result_ref_tu)

