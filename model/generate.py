import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# 加载训练后的BART模型和tokenizer
model_name = "./model/finetuned_bart"  # 替换为你的模型路径
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def generate_response(received_email, subject=None, category=None):
    """
    根据接收的邮件和类别生成回复邮件。

    参数:
    received_email (str): 接收的邮件内容。
    category (str): 邮件类别。

    返回:
    str: 生成的回复邮件。
    """
    # 构建输入文本
    if subject is None and category is None:
        input_text = f"<|received|>{received_email}<|reply|>"
    elif subject is None and category is not None:
        input_text = f"<|category|>{category}<|received|>{received_email}<|reply|>"
    elif subject is not None and category is None:
        input_text = f"<|received|>{received_email}<|subject|>{subject}<|reply|>"
    else:
        input_text = f"<|category|>{category}<|received|>{received_email}<|subject|>{subject}<|reply|>"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # 生成回复
    output = model.generate(
        input_ids,
        max_length=400,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    # 解码生成的回复
    response = tokenizer.decode(output[0], skip_special_tokens=True)    
    return response


if __name__ == '__main__':
    # 示例用法
    received_email = """
    Dear Admissions Committee,

    I hope this email finds you well. My name is Sarah Johnson, and I am writing to express my strong interest in applying for the Master's program in Computer Science at your esteemed university. I have completed my undergraduate degree in Computer Science at SJTU, and I believe my academic background and passion for research make me a strong candidate for your program.

    I have attached my resume, transcripts, and letters of recommendation for your review. Please let me know if you need any further information or materials.

    Thank you for your time and consideration. I look forward to hearing from you.

    Best regards,
    Sarah Johnson

    """
    response = generate_response(received_email, category="customer service inquiry")
    print("Generated Response:", response)