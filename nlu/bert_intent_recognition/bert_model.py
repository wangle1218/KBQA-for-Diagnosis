#! -*- coding: utf-8 -*-
from bert4keras.backend import keras,set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam

set_gelu('tanh')

def textcnn(inputs,kernel_initializer):
	# 3,4,5
	cnn1 = keras.layers.Conv1D(
			256,
			3,
			strides=1,
			padding='same',
			activation='relu',
			kernel_initializer=kernel_initializer
		)(inputs) # shape=[batch_size,maxlen-2,256]
	cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)  # shape=[batch_size,256]

	cnn2 = keras.layers.Conv1D(
			256,
			4,
			strides=1,
			padding='same',
			activation='relu',
			kernel_initializer=kernel_initializer
		)(inputs)
	cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)

	cnn3 = keras.layers.Conv1D(
			256,
			5,
			strides=1,
			padding='same',
			kernel_initializer=kernel_initializer
		)(inputs)
	cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)

	output = keras.layers.concatenate(
		[cnn1,cnn2,cnn3],
		axis=-1)
	output = keras.layers.Dropout(0.2)(output)
	return output

def build_bert_model(config_path,checkpoint_path,class_nums):
	bert = build_transformer_model(
		config_path=config_path, 
		checkpoint_path=checkpoint_path, 
		model='bert', 
		return_keras_model=False)

	cls_features = keras.layers.Lambda(
		lambda x:x[:,0],
		name='cls-token'
		)(bert.model.output) #shape=[batch_size,768]
	all_token_embedding = keras.layers.Lambda(
		lambda x:x[:,1:-1],
		name='all-token'
		)(bert.model.output) #shape=[batch_size,maxlen-2,768]

	cnn_features = textcnn(
		all_token_embedding,bert.initializer) #shape=[batch_size,cnn_output_dim]
	concat_features = keras.layers.concatenate(
		[cls_features,cnn_features],
		axis=-1)

	dense = keras.layers.Dense(
			units=512,
			activation='relu',
			kernel_initializer=bert.initializer
		)(concat_features)

	output = keras.layers.Dense(
			units=class_nums,
			activation='softmax',
			kernel_initializer=bert.initializer
		)(dense)

	model = keras.models.Model(bert.model.input,output)

	return model

if __name__ == '__main__':
	config_path='E:/bert_weight_files/bert_wwm/bert_config.json'
	checkpoint_path='E:/bert_weight_files/bert_wwm/bert_model.ckpt'
	class_nums=13
	build_bert_model(config_path, checkpoint_path, class_nums)