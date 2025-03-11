import telebot
import cv2
import numpy as np
import bot_secrets
import logging
import io
from transformationImg import transform

logging.basicConfig(
    format="[%(levelname)s %(asctime)s %(module)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

bot = telebot.TeleBot(bot_secrets.TOKEN)


@bot.message_handler(commands=['start'])
def send_requirements(message):
    """Send the explanation and requirements when the user starts."""
    explanation = """
Welcome! This bot processes images and extracts documents.
Please make sure the following requirements are met before uploading an image:

1. The background of the document should be uniform when the document is the central object in the image.
2. The document should be visible and well-defined, with clear contours.
3. The bot will extract and process the document area.
4. A dark background is recommended.

After you've ensured your image meets these criteria, feel free to send it over and I’ll process it for you.
    """
    bot.reply_to(message, explanation)



@bot.message_handler(content_types=["photo"])
def handle_photo(message: telebot.types.Message):
    logger.info(f"- received photo from {message.chat.username}")

    # Get the photo in the highest quality
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)

    # Download the file to memory (RAM)
    downloaded_file = bot.download_file(file_info.file_path)

    # Convert the image to a NumPy array
    np_arr = np.frombuffer(downloaded_file, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    processed_image = transform(image)
    if processed_image is None:  # If no contours or error occurred
        bot.reply_to(message, "No contours detected or an error occurred.")
        return

    # Compress the image to JPEG format and save in memory
    _, img_encoded = cv2.imencode('.jpg', processed_image)
    img_byte_arr = io.BytesIO(img_encoded.tobytes())

    # Send the processed image back to the user
    bot.send_photo(message.chat.id, img_byte_arr, caption="Here is your processed document!")

    logger.info("- processed image sent successfully")


@bot.message_handler(func=lambda x:True)
def handle_photo(message: telebot.types.Message):
    bot.reply_to(message, 'Please uploading an image For instructions write /start')




logger.info("* starting bot")
bot.infinity_polling()
logger.info("* goodbye!")