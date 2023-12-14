import pandas as pd
import cohere
import numpy as np
df=pd.read_csv('train.csv')
#drop all nan
df=df.dropna()
q1arr=[x for x in df['question1'].values]
q2arr=[x for x in df['question2'].values]
print(len(q1arr))

def split_array_into_batches(arr, batch_size):
    batches = []
    for i in range(0, len(arr), batch_size):
        batches.append(arr[i:i + batch_size])
    return batches

arrs=split_array_into_batches(q1arr,1500)
q2arrs=split_array_into_batches(q2arr,1500)

co = cohere.Client('xxx') # This is your API key

q1_embeddings = []  # List to store embeddings from all batches

for i in range(len(arrs)):
    print(f"Processing batch {i}")
    response = co.embed(
        model='embed-english-light-v3.0',
        texts=arrs[i],
        input_type='classification'
    )
    embeddings = response.embeddings

    # Add the embeddings of the current batch to the list
    if i==0:
        q1_embeddings=np.array(embeddings)
    else:
        q1_embeddings = np.append(q1_embeddings, embeddings, axis=0)

    # Save the updated list of embeddings every once in a while
    if i % 30 == 0:
        np.save('q1CohereEmbedLight', q1_embeddings)
np.save('q1CohereEmbedLight', q1_embeddings)
q2_embeddings = []  # List to store embeddings from all batches
for i in range(len(q2arrs)):
    print(f"Processing batch {i}")
    response = co.embed(
        model='embed-english-light-v3.0',
        texts=q2arrs[i],
        input_type='classification'
    )
    embeddings = response.embeddings

    if i==0:
        q2_embeddings=np.array(embeddings)
    else:
        q2_embeddings = np.append(q2_embeddings, embeddings, axis=0)

    # Save the updated list of embeddings every once in a while
    if i % 30 == 0:
        np.save('q2CohereEmbedLight', q2_embeddings)
np.save('q2CohereEmbedLight', q2_embeddings)

print(len(q1_embeddings))
embeddings_q1_df=pd.DataFrame(q1_embeddings, dtype=np.float32)
embeddings_q1_df.columns=[f'q1_emb_{i}' for i in range(384)]
print(len(embeddings_q1_df))


print(len(q2_embeddings))
embeddings_q2_df=pd.DataFrame(q2_embeddings, dtype=np.float32)
embeddings_q2_df.columns=[f'q2_emb_{i}' for i in range(384)]
print(len(embeddings_q2_df))

df=pd.read_csv('train.csv')
#drop all nan
df=df.dropna()
print(len(df))
df.reset_index(drop=True, inplace=True)

dedf=pd.concat([df, embeddings_q1_df, embeddings_q2_df], axis=1)
len(dedf)

dedf.drop(['question1', 'question2', 'qid1', 'qid2', 'id'], axis=1, inplace=True)
dedf.head(20)
dedf.to_csv('train_embedded_light.csv', index=False)