import argparse
import smtplib
from email.message import EmailMessage

import yaml


def send_email(subject, body, config_path="./configs/email_config.yml"):
    # 读取邮件配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['email']

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = config['from']
    msg['To'] = config['to']
    msg.set_content(body)

    # 发送邮件
    try:
        with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as smtp:
            smtp.starttls()
            smtp.login(config['from'], config['password'])
            smtp.send_message(msg)
        print("[✓] Notification email sent successfully.")
    except Exception as e:
        print(f"[✗] Failed to send notification email: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send Email Notification")
    parser.add_argument("--subject", type=str, required=True, help="Email Subject")
    parser.add_argument("--body", type=str, required=True, help="Email Body")
    args = parser.parse_args()

    send_email(args.subject, args.body)
