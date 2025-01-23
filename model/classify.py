import joblib

class EmailClassifier:
    def __init__(self):
        self.classifier, self.vectorizer, self.classifier10, self.vectorizer10 = self.load_classifier()

    def load_classifier(self):
        # 加载模型和向量化器
        model_path = 'model/collection/spam_classifier_model.pkl'
        vectorizer_path = 'model/collection/vectorizer.pkl'
        classifier = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        # 加载模型和向量化器 10class
        classifier10 = joblib.load('model/collection/email_classifier_model.pkl')
        vectorizer10 = joblib.load('model/collection/tfidf_vectorizer.pkl')

        return classifier, vectorizer, classifier10, vectorizer10

    def predict_spam(self, message):
        message_counts = self.vectorizer.transform([message])
        prediction = self.classifier.predict(message_counts)
        return 'spam' if prediction[0] == 1 else 'ham'

    def predict_email_category(self, email_subject, email_body):
        email_text = email_subject + " " + email_body
        email_vector = self.vectorizer10.transform([email_text])
        prediction = self.classifier10.predict(email_vector)
        return prediction[0]

    def classify(self, email_subject, email_body):
        if self.predict_spam(email_body) == "spam":
            return "spam"
        else:
            return self.predict_email_category(email_subject, email_body)

if __name__ == '__main__':
    # 使用示例
    email_classifier = EmailClassifier()
    result = email_classifier.classify("", "! We've won a free ticket to HongKong. !")
    print(result)