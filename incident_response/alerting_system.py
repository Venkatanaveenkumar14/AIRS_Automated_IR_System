import smtplib
from email.mime.text import MIMEText

def send_alert(subject, message, recipient):
    """
    Sends an email alert to the specified recipient.
    """
    sender_email = "noreply@securitysystem.com"
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient

    with smtplib.SMTP('localhost') as server:
        server.sendmail(sender_email, recipient, msg.as_string())
    print(f"Alert sent to {recipient}")