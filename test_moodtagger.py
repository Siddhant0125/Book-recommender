from mood_classifier_tagger.mood_tagger import MoodTagger

tagger = MoodTagger()
print(tagger.tag("I feel anxious but want something hopeful and cozy", top_k=5))
print(tagger.tag("Happy romance", top_k=5))
