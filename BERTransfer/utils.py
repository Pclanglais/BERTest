#The function to create the BERTopic model.
#The main difference with the official BERTopic implementation is that we return also the document embeddings that will be necessary for further calculations
def create_bertopic(docs, language = "english", calculate_probabilities=True, verbose=True, bert_model = "all-MiniLM-L6-v2", similarity_threshold = 0.01, document_selection = 20):
  from bertopic import BERTopic
  from sentence_transformers import SentenceTransformer
  from BERTransfer import BERTopicM

  sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
  embeddings = sentence_model.encode(docs, show_progress_bar=False)

  # Train our topic model using our pre-trained sentence-transformers embeddings
  topic_model = BERTopic(language=language, calculate_probabilities=True, verbose=True)
  topics, probs = topic_model.fit_transform(docs, embeddings)
  bertopic_model = BERTopicM(topic_model = topic_model, topics = topics, probs = probs, embeddings = embeddings, similarity_threshold = similarity_threshold, document_selection = document_selection)
  return bertopic_model
