from argparse import ArgumentParser, Namespace
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from redis import Redis
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField
from redis.commands.search.field import TagField
from redis.commands.search.query import Query
from redis.commands.search.result import Result

# cli app initiation  
parser = ArgumentParser()


# usage message
parser.usage = "py similarity_que_nlp.py 'type your question' 5\n It will print top 5 similar question as input question."


data_path = 'Data\database.csv'
# data_path = 'Data\product_data.csv'

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


host = 'localhost'
port = 6379
redis_conn = Redis(host = host, port = port)
if redis_conn:
    print ('Connected to redis')


MAX_TEXT_LENGTH = 512
NUMBER_QUE = 10001

# define a function to auto-truncate long text fields
def auto_truncate(val):
    return val[:MAX_TEXT_LENGTH]


# load the product data and truncate long text fields
all_que_df = pd.read_csv(data_path, converters={'bullet_point': auto_truncate, 'item_keywords': auto_truncate, 'item_name': auto_truncate})

# print(all_que_df.info())
all_que_df.dropna(subset=['question'], inplace=True)
all_que_df.reset_index(drop=True, inplace=True)


# get the first 1000 products with non-empty item keywords
que_metadata = all_que_df.to_dict(orient='index')

model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

que_keywords =  [que_metadata[i]['question']  for i in que_metadata.keys()]
que_keywords_vectors = [ model.encode(sentence) for sentence in que_keywords]

def load_vectors(client:Redis, df_metadata, vector_dict, vector_field_name):
    p = client.pipeline(transaction=False)
    for index in df_metadata.keys():    
        #hash key
        key='que:'+ str(index)
        
        #hash values
        que_metadata = df_metadata[index]
        que_keywords_vector = vector_dict[index].astype(np.float32).tobytes()
        que_metadata[vector_field_name]=que_keywords_vector
        
        # HSET
        p.hset(key,mapping=que_metadata)        
    p.execute()

def create_flat_index (redis_conn,vector_field_name,number_of_vectors, vector_dimensions=512, distance_metric='L2'):
    redis_conn.ft().create_index([
        VectorField(vector_field_name, "FLAT", {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric, "INITIAL_CAP": number_of_vectors, "BLOCK_SIZE":number_of_vectors }),
        TextField("question"),
        TextField("text"),       
    ])

ITEM_KEYWORD_EMBEDDING_FIELD='que_keyword_vector'
TEXT_EMBEDDING_DIMENSION=768
NUMBER_QUE=10001

print ('Loading and Indexing + ' +  str(NUMBER_QUE) + ' questions')

#flush all data
redis_conn.flushall()

#create flat index & load vectors
create_flat_index(redis_conn, ITEM_KEYWORD_EMBEDDING_FIELD,NUMBER_QUE,TEXT_EMBEDDING_DIMENSION,'COSINE')
load_vectors(redis_conn,que_metadata,que_keywords_vectors,ITEM_KEYWORD_EMBEDDING_FIELD)



parser = ArgumentParser()


# usage message
parser.usage = "py similarity.py 'type your question' 5\n It will print top 5 similar question as input question."
parser.add_argument('question', help="Input the question to find similarity", type=str)
parser.add_argument('limit', help="total number of question's (LIMIT)", type=int, default=5)

args:Namespace = parser.parse_args()
def get_similar_que(args):
    print(f"Your question : {args.question}, with LIMIT: {args.limit}")
    topK = args.limit
    que_query= args.question
    #product_query='cool way to pimp up my cell'

    #vectorize the query
    query_vector = model.encode(que_query).astype(np.float32).tobytes()

    #prepare the query
    q = Query(f'*=>[KNN {topK} @{ITEM_KEYWORD_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score').paging(0,topK).return_fields('vector_score','question').dialect(2)
    params_dict = {"vec_param": query_vector}


    #Execute the query
    results = redis_conn.ft().search(q, query_params = params_dict)

    #Print similar products found
    for question in results.docs:
        print ('***************question  found ************')
        print (color.BOLD + 'hash key = ' +  color.BOLD + question.id)
        print (color.GREEN + 'question = ' +  color.GREEN  + question.question)
        print (color.YELLOW + 'Score = ' +  color.YELLOW  + question.vector_score)

get_similar_que(args)