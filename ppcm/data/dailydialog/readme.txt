Here are some explanations about the files:

1) dialogues_text.txt: The DailyDialog dataset which contains 11,318 transcribed dialogues.
2) dialogues_topic.txt: Each line in dialogues_topic.txt corresponds to the topic of that in dialogues_text.txt.
                        The topic number represents: {1: Ordinary Life, 2: School Life, 3: Culture & Education,
                        4: Attitude & Emotion, 5: Relationship, 6: Tourism , 7: Health, 8: Work, 9: Politics, 10: Finance}
3) dialogues_act.txt: Each line in dialogues_act.txt corresponds to the dialog act annotations in dialogues_text.txt.
                      The dialog act number represents: { 1: inform，2: question, 3: directive, 4: commissive }
4) dialogues_emotion.txt: Each line in dialogues_emotion.txt corresponds to the emotion annotations in dialogues_text.txt.
                          The emotion number represents: { 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise}
5) train.zip, validation.zip and test.zip are two different segmentations of the whole dataset. 

EMO
Performance on test set: Average loss: 0.3928, Accuracy: 8873/10295 (86%)
Epoch took: 2.351s

Example prediction
Input sentence: This is incredible! I love it, this is the best chicken I have ever had.
Predictions: no_emotion: 0.0175, anger: 0.0012, disgust: 0.0006, fear: 0.0000, happiness: 0.9742, sadness: 0.0003, surprise: 0.0061

Minimum loss on test set obtained at epoch 49
Maximum accuracy on test set obtained at epoch 53


ACT
Performance on test set: Average loss: 0.5387, Accuracy: 8184/10295 (79%)
Epoch took: 1.782s

Example prediction
Input sentence: This is incredible! I love it, this is the best chicken I have ever had.
Predictions: inform: 0.8892, question: 0.0119, directive: 0.0200, commissive: 0.0790

Minimum loss on test set obtained at epoch 85
Maximum accuracy on test set obtained at epoch 62


TOPICS
Performance on test set: Average loss: 1.2817, Accuracy: 5692/10296 (55%)
Epoch took: 1.974s

Example prediction
Input sentence: This is incredible! I love it, this is the best chicken I have ever had.
Predictions: Ordinary Life: 0.6156, School Life: 0.0007, Culture & Education: 0.0046, Attitude & Emotion: 0.1055, Relationship: 0.2416, Tourism: 0.0048, Health: 0.0031, Work: 0.0220, Politics: 0.0002, Finance: 0.0019

Minimum loss on test set obtained at epoch 85
Maximum accuracy on test set obtained at epoch 45




EMOCAP
Performance on test set: Average loss: 0.4606, Accuracy: 979/1150 (85%)
Epoch took: 2.888s

Example prediction
Input sentence: This is incredible! I love it, this is the best chicken I have ever had.
Predictions: others: 0.0719, happy: 0.9016, sad: 0.0083, angry: 0.0183

Minimum loss on test set obtained at epoch 80
Maximum accuracy on test set obtained at epoch 1