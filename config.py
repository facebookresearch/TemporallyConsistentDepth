
znear = 0.1
zfar = 1e3 #15 #25 #15

train_data_file = 'fusiondata_train.csv' 
val_data_file = 'fusiondata_val.csv'

batch_size = 18 #6 #18 #18 #6 
num_gpus = 6
learning_rate = 1e-4 # Replica finetuning DPT 5e-4 #1e-6 #3.5e-4 
val_interval = 5 
save_interval = 100
accum_interval = 1
num_epochs = 1000
validation_split = 0.05
sequence_len = 15 #10
tstable = 0.15 #0.25

sequence_len_test = 1 #20 # TODO: Undo 59 
batch_size_test = 1

max_w = 5

# check all ATTN!! in code
