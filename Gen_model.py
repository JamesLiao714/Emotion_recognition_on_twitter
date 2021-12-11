import pandas as pd
import numpy as np
import time
import datetime
import preprocessor.api as p # needs tweet-preprocessor package
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, random_split
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np
import os
import regex as re
from dataset import TweetDataset
#config
# REGEN:
# True: train a new model 
# False: train our saved pretrained model

REGEN = False
HT = False #concat hashtag
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = './data/'
#REGEN model
# "vinai/bertweet-base"
pretrain_model = 'roberta-base'
# my pretrained model
pretrain_dir = './model_save/roberta_lite_pre/'
output_dir = './model_save/roberta_lite_ep4/'
batch_size = 16

print('Using device: ', device)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# data cleaning 
def preprocess_tweets(df):
    # convert to lower case
    df['text'] = df.text.str.lower()
    # remove links
    df.text = df.text.apply(lambda x: re.sub(r'https?:\/\/\S+', 'http', x))
    df.text = df.text.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))
    df.text = df.text.apply(lambda x: re.sub(r'{link}', '', x))
    df.text = df.text.apply(lambda x: re.sub(r"\[video\]", '', x))
    # substitute 'RT @' with '@'
    df.text = df.text.apply(lambda x: re.compile('rt @').sub('@', x).strip())
    # Remove usernames. The usernames are any word that starts with @.
    df.text = df.text.apply(lambda x: re.sub('\@[a-zA-Z0-9]*', '@user', x))
    # convert '#' to '' and '_' to ' ' and ':' to ''
    df.text = df.text.apply(lambda x: x.replace("<lh>", ''))
    return df

#Load our data
print('LOADING DATA')
data_id = pd.read_csv(data_path + 'data_identification.csv')
emotions = pd.read_csv(data_path + 'emotion.csv')
sample_sub = pd.read_csv(data_path + 'sampleSubmission.csv')
tweets = pd.read_json(data_path + 'tweets_DM.json', lines=True)
tweets_important = pd.DataFrame(tweets._source)
print('FINISHED DATA LOADING')


# Create a final dataframe from all of the data source
tw_list = tweets_important['_source'].to_list()
tmp_df = pd.DataFrame.from_records(tw_list)
tmp_df_list = tmp_df['tweet'].to_list()
final_tweet_df = pd.DataFrame.from_records(tmp_df_list)
df_final = pd.merge(final_tweet_df, data_id, how='outer', on='tweet_id').merge(emotions, how='outer', on='tweet_id')

# clean our data
print('DATA CLEANING PROCEEDING')
# df_final['text'] = df_final['text'].str.lower().str.replace('\s\s+', ' ').str.replace('<lh>', '').str.strip()
df_final  = preprocess_tweets(df_final)
print(df_final.head())

if HT:
    #concat the hash tag NOT NECESSARY
    print('HASH TAG CONCAT')
    df_final['hashtags'] = df_final.hashtags.apply(lambda t: ' '.join(t).lower())
    #add hashtag to text for training
    df_final['text'] = df_final['text'] + ' ' + df_final['hashtags']


# Separate training from testing data
train_df = df_final[df_final['identification'] == 'train']
test_df = df_final[df_final['identification'] == 'test']

# How big are the training and testing data
print('TRAIN DATA shape:', train_df.shape)
print('TEST DATA shape:', test_df.shape)
print(df_final['text'][:10])
print('START BERT')

if REGEN:
    print('using tokenizer: ', pretrain_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model, use_fast=False)
else:
    print('using tokenizer: ', pretrain_dir)
    tokenizer = AutoTokenizer.from_pretrained(pretrain_dir, use_fast=False)
# tweet dataset for training and validatingｂ
dataset = TweetDataset(train_df, 'train', tokenizer)

# 95% training, 5% validation since the training data is large
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# train dataloader 
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True) 


# validation dataloader
validation_dataloader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
        
if REGEN:
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = RobertaForSequenceClassification.from_pretrained(
        pretrain_model, # Use model select
        num_labels = 8, # The number of output labels
        output_attentions = False, # returns attentions weights.
        output_hidden_states = False, # rereturns all hidden-states.
    )
    model.cuda()
else:
    model = RobertaForSequenceClassification.from_pretrained(pretrain_dir)
    model.to(device)

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

# Number of training epochs. The BERT authors recommend between 2 and 4. 
epochs = 2

# Total number of training steps is [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
                    


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    #               Training
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()
    # Reset the total loss
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # reference : https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    model.train()

    # Batch
    for step, batch in enumerate(train_dataloader):

        # Progress update every 100 batches.
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # [0]: input ids 
        # [1]: attention masks
        # [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a backward pass. 
        # Pytorch accumulates the gradients is "convenient while training RNNs". 
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # Return 
        #  (1)loss
        #  (2)logits
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels,
                             return_dict = False)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # this is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    print(" Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
       
    #               Validation
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():        
 
            # The "logits" are the output values prior to applying an activation function like the softmax.
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels,
                                   return_dict = False)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )


print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

