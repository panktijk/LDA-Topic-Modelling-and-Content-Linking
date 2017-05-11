# -*- coding: utf-8 -*-
from nltk.tokenize import RegexpTokenizer
import stop_words
import gensim
from gensim import corpora, models, similarities
from operator import itemgetter
import numpy as np
from sklearn.metrics import mutual_info_score

def get_sections(doc):
    with open(doc, 'r') as chapter:
        content = chapter.read().decode('utf-8')
        subsections = content.split('\n\n')
    return [section.replace('\n', ' ') for section in subsections]

def get_tokens(sections):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    en_stop_words = stop_words.get_stop_words('en')
    tokens = []
    for section in sections:
        raw_doc = section.lower()
        raw_tokens = tokenizer.tokenize(raw_doc)
        stemmed_tokens = [token for token in raw_tokens if (len(token) > 1) & (token not in en_stop_words)]
        tokens.append(stemmed_tokens)
    return tokens
        
def create_corpus(tokens):
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    return corpus, dictionary

def build_lda_model(corpus, dictionary, num_passes, num_topics_gen, num_topics_show, num_words_per_topic):	
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics_gen, id2word=dictionary, passes=num_passes)
    all_topics = ldamodel.show_topics(num_topics=num_topics_show, num_words=num_words_per_topic, formatted=True)
    for topic in all_topics:
        print topic
    return ldamodel

def get_topics_per_section(ldamodel, corpus, min_prob):
    section_topics = {}
    for section in corpus:
        topics = ldamodel.get_document_topics(section, minimum_probability=min_prob, minimum_phi_value=None, per_word_topics=False)
        section_topics[corpus.index(section)] = max(topics, key=itemgetter(1))[0]
    return section_topics
    
def get_dist_for_topic(ldamodel, topic_id):
    term_probs = ldamodel.get_topic_terms(topic_id)
    terms, probs = zip(*term_probs)
    return list(terms), list(probs)

def get_similar_sections(sim_matrix, all_topics, section_id, threshold):
    section = all_topics[section_id]
    return [np.where(sim_matrix[section] == s_id) for s_id in sim_matrix[section] if s_id > threshold]

def map_similar_sections(sim_matrix, all_topics, num_sections, threshold):
    sections_map = {}
    for i in range(num_sections):
        sim_sections = get_similar_sections(sim_matrix, all_topics, i, threshold)
        sections_map[i] = [sim_section[0] for sim_section in sim_sections]
    for sec_id, sim_sec_ids in sections_map.iteritems():
        sim_sections = []
        for sections in sim_sec_ids:
            for sec in sections:
                #if (sec != sec_id) & (sec not in sim_sections):
                #    print sec, sec_id, sim_sections
                    sim_sections.append(sec)
        sections_map[sec_id] = sim_sections
    return sections_map
    
def create_section_topic_matrix(all_topics):
    section_topics = []
    for topic in all_topics:
        row = []
        for i in range(0, 50):
            if i in [t[0] for t in topic]:
                row.append([x[1] for x in topic if (x[0] == i)][0])
            else:
                row.append(0)
        section_topics.append(row)
    return section_topics
    
def create_mutual_info_matrix(section_topics):
    mi_matrix = []
    for topic1 in section_topics:
        mi_for_topic = []
        for topic2 in section_topics:
            mi_for_topic.append(mutual_info_score(topic1, topic2))
        mi_matrix.append(mi_for_topic)
    return mi_matrix
    
def get_similar_sections_mi(mi_matrix, threshold):
    sections_map = {}
    for row in range(0,len(mi_matrix)):
        sections_map[row] = []
        for sec in range(0, len(mi_matrix)):
            if((row != sec) & (mi_matrix[row][sec] > threshold)):
                (sections_map[row]).append(sec)
    return sections_map
            
        
        
sections1 = get_sections('sample1.txt')
sections2 = get_sections('sample2.txt')
sections = sections1 + sections2
tokens1 = get_tokens(sections1)
tokens2 = get_tokens(sections2)
tokens = tokens1+tokens2
corpus, dictionary = create_corpus(tokens)
ldamodel = build_lda_model(corpus, dictionary, 100, 50, 50, 5)
section_topics = get_topics_per_section(ldamodel, corpus, None)
all_topics = ldamodel[corpus]
sim_matrix = similarities.MatrixSimilarity(all_topics)
#print get_similar_sections(sim_matrix, all_topics, 34)
similar_sections_cos = map_similar_sections(sim_matrix, all_topics, len(sections1), 0.3)
sections_topic_matrix = create_section_topic_matrix(all_topics)
mi_matrix = create_mutual_info_matrix(sections_topic_matrix)
similar_sections_mi = get_similar_sections_mi(mi_matrix, 0.0001)
for (k, v) in similar_sections_cos:
    print (k, v)
#for (k, v) in similar_sections_mi:
    #print (k, v)

    
    
