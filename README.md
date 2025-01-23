# Smart Email Processing System  

## Overview  
This project is a **Smart Email Processing System** designed to enhance the email experience by intelligently managing and automating key tasks. It combines machine learning and natural language processing techniques to classify emails and generate context-aware replies.  

## Features  
### 1. **Spam Email Classification**  
- Implemented using a **Naive Bayes classifier**.  
- Accurately categorizes incoming emails into **spam** and other normal categories.  

### 2. **Email Reply Generation**  
- Powered by **BART (Bidirectional and Auto-Regressive Transformers)**.  
- Generates fluent and contextually relevant replies to user-selected emails based on daily conversations.  

### 3. **User-Friendly Interface**  
- A custom-built **Graphical User Interface (GUI)** for seamless interaction.  
- Users can:  
  - View emails and their categories.  
  - Generate replies to emails.  
  - Delete or mark emails as spam directly in the interface.  


![image](assets/mailAPP)


## Installation  
### Prerequisites  
- Python 3.8+  


### Steps  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/AdaLovelemon/EmailAgent.git  
   cd EmailAgent
   ```  
2. Run the application:  
   ```bash  
   python mail.py  
   ```  

## How It Works  
### 1. **Spam Classification**  
- **Model**: Multinomial Naive Bayes.  
- **Input**: Preprocessed email text.  
- **Output**: A label.  
- **Training Data**: Publicly available email datasets like Enron or SpamAssassin.  

### 2. **Reply Generation**  
- **Model**: BART pre-trained transformer (fine-tuned on email datasets for contextual reply generation).  
- **Input**: Email body text.  
- **Output**: A coherent and relevant reply.  

### 3. **User Interface**  
- Built with **Tkinter** for a lightweight and responsive experience.  
- Includes features like email list display, reply generation button, and spam management options.  

## Project Highlights  
- **Hybrid Approach**: Combines traditional ML (Naive Bayes) and modern NLP (BART).  
- **Efficiency**: Quick spam detection with Naive Bayes and state-of-the-art text generation with BART.  
- **Usability**: Intuitive UI designed for non-technical users.  
- **Real-time Email Integration**: Connect the system to live email accounts (e.g., Gmail, Outlook).  

## Future Work  
- **Improved Spam Filtering**: Incorporate ensemble models for higher accuracy.  
- **Multi-language Support**: Expand email processing capabilities to support multiple languages.  

## Credits  
- **Team Members**: **Junhang He**, **Hongming Guo**, **Qizhi Zhang**, **Ruixuan Zhu** 
- **Acknowledgments**: Special thanks to **IAIR** of **XJTU**.

## License  
This project is licensed under the MIT License.  
