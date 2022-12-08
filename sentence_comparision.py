from difflib import *
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch
import nltk

def opcodes(a, b):
    s = SequenceMatcher(None, a, b)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        print('{:7}   query[{}:{}] --> matched[{}:{}] {!r:>8} --> {!r}'.format(tag, i1, i2, j1, j2, a[i1:i2], b[j1:j2]))
        
        
def ndiff(a,b):
    print("".join(ndiff(a.splitlines(keepends=True), b.splitlines(keepends=True))), end="")
    
def syntactic_ratio(a, b):
    # match sentence a and b
    s = SequenceMatcher(lambda x: x == " ", a, b)
    
    # return their ratio
    # rule-of-thumb: ratio > 0.6 -> similar
    return s.ratio()

def find_additions_deletions(a, b):
    """
    finds the differnce inform of
    two strings addition/ deletions
    between two strings
    """
    
    # init differ
    d = Differ()
    
    # compare the two 
    diff = d.compare(a, b)
    changes = [change for change in diff if change.startswith('-') or  change.startswith('+')]
    
    # output:
    additions = ""
    deletions = ""
    
    for change in changes:
        type_of_change  = 'addition' if change[0] == '+' else 'deletion'
        
        # remove unwanted symbols
        actual_change = change[2:]
        
        if type_of_change == 'addition':
            additions += actual_change
            
        else:
            deletions += actual_change
    
    return additions, deletions


def match_sentences(document_a, document_b, k = 1, model='all-MiniLM-L6-v2', threshold=0.6):
    
    # Model to be used to create Embeddings which we will use for semantic search
    embedder = SentenceTransformer(model)
    
    # Use the sentences in A as queries
    queries = nltk.sent_tokenize(document_a)
    
    # Use the sentences in B as our corpus
    corpus = nltk.sent_tokenize(document_b)
    
    # Create embeddings using B
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # matched_sentences dict:
    # key = query_idx
    # value = list of matched sentences and score pairs = [(matched_sentence, similarity_score)]
    matched_sentences = {i:[] for i in range(len(queries))}
    
    # Find the closest k most similar sentences using cosine similarity
    top_k = min(k, len(corpus))
    for query_idx in range(len(queries)):
        query = queries[query_idx]
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest k scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        
        # loop over top results
        for score, idx in zip(top_results[0], top_results[1]):
            
            # fill the matched sentences dictonary
            #if score > threshold:
            matched_sentences[query_idx].append((idx, score))
    
    return matched_sentences

def find_added_indices(matched_indices, corpus_length):
    corpus_indices = list(range(corpus_length))
    
    return list(set(corpus_indices) - set(matched_indices))

def detect_changes(matched_dict, document_a, document_b, important_indices, top_k=1, show_output=False):
    
    # Use the sentences in A as queries
    queries = nltk.sent_tokenize(document_a)
    
    changed_sentences = []
    
    corpus = nltk.sent_tokenize(document_b)
    
    matched_indices = []
    
    save_additions = {}
    
    save_deletions = {}
    
    for query_idx in range(len(queries)):
        
        # current query
        query = queries[query_idx]
        
        # give lower bound on number of matched sentences
        top_k = min(top_k, len(matched_dict[query_idx]))
        
        
        
        
        for k in range(top_k):
            
            # get current matched_sentence + score
            matched_idx, score = matched_dict[query_idx][k]
            
            matched_indices.append(int(matched_idx))
            
            # get the actual sentence
            matched_sentence = corpus[int(matched_idx)]
            
            
            additions, deletions = find_additions_deletions(query, matched_sentence)
            
            # get syntactic ratio
            ratio = syntactic_ratio(query, matched_sentence)
            
            if show_output:
                print(f"query: {query}\nmatched: {matched_sentence}\nSemantic Resemblence: {score:.4f}\n"
                      f"Syntactic Resemblence: {ratio:.4f}\n")

                # extract addtions and deletions
                
                
                print(f"added in newer version:{additions}\ndeleted from older version: {deletions}")
                
                print("------------------------------------------------------------------------------\n")

            if ratio < 1.0:
                changed_sentences.append(query_idx)
                
                save_additions[query_idx] = additions
                
                save_deletions[query_idx] = deletions
                
    #drop_unimportant_indices(changed_sentences, important_indices=important_indices[version])
    
    
    new_sentences = find_added_indices(matched_indices, len(corpus))

    return changed_sentences, new_sentences, save_additions, save_deletions
                    
            
                    
def calculate_change_importance(changed_idx, matched_dict, ranking,
                                threshold, version,w0, w1, w2, top_k =1):
    
    """Calculate how important an actual change was.
    
    Using Hyp 1-2

    Args:
       matched_dict: calcualted using match_sentences
       ranking: sentence_importance obtained using rank_YAKE!
       threshold: Hyperparameter to determine if sentence is new
       w0-w2: optionally tuned weights
       top_k: only usefull if sentence matching is being ran with extra k
        
    Returns:
        Importance of 1 change.
    """
    
    
    for k in range(top_k):
        matched_idx, score = matched_dict[changed_idx][k]


        I_s = ranking[version][changed_idx]

        next_I_s = ranking[version + 1][int(matched_idx)]

        if score < threshold:
            # Hypothesis 2
            I_c = next_I_s * (w2/ w1 * score)
        else:    
            # Hypothesis 1
            I_c = I_s * (w0/ w1 * score)
    
    return I_c


def calculate_change_importances(changed_indices, matched_dict,
                                 ranking, threshold, version, w0=1, w1=1, w2=1):
    
    return {changed_index: calculate_change_importance(changed_index, matched_dict,ranking,threshold, version, w0, w1, w2) 
            for changed_index in changed_indices}







