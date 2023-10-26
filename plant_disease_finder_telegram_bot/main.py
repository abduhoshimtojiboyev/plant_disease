import logging
import random

import cv2
import numpy as np
from io import BytesIO
import os
import telegram
from PIL import Image
from typing import Final
from transformer import MyTransform

from PIL.Image import Transform
from fastai.learner import load_learner
from fastai import *
from fastai.vision.core import PILImage
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters
)

import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
learn = load_learner(fname='model/model.pkl')
pathlib.PosixPath = temp

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Stages
START_ROUTES, END_ROUTES = range(2)
# Callback data
ONE, TWO, THREE, FOUR, FIVE = range(5)
TYPES = None


# choose_category, contact us, Tomato,potato,pepper

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send message on `/start`."""
    user = update.message.from_user
    logger.info("User %s started the conversation.", user.first_name)

    keyboard = [
        [
            InlineKeyboardButton("Choose Category", callback_data=str(category)),
            InlineKeyboardButton("Contact Us", callback_data=str(contact)),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    # Send message with text and appended InlineKeyboard
    await update.message.reply_text(f"Hi {user.first_name}!, I am a plant disease finder bot. I have been trained to "
                                    f"find how your plant have disease or not based on the picture of the leaf of the "
                                    f"plant . If it has disease which type of"
                                    f"disease it is. Currently we only work with 'Tomato', 'Potato' and 'Pepper'."
                                    f"If Your are not sure with bot choose the 'Contact' button",
                                    reply_markup=reply_markup)
    # Tell ConversationHandler that we're in state `FIRST` now
    return START_ROUTES


async def start_over(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    keyboard = [
        [
            InlineKeyboardButton("Choose Category", callback_data=str(category)),
            InlineKeyboardButton("Contact Us", callback_data=str(contact)),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.reply_text(text="Start handler, Choose a route", reply_markup=reply_markup)
    return START_ROUTES


async def category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show new choice of buttons"""
    query = update.callback_query
    examples_list = ['images/new_1.jpeg', 'images/new_2.jpeg', 'images/new_3.jpg', 'images/new_4.jpg',
                     'images/new_5.jpg'
        , 'images/new_6.png', 'images/new_7.jpg']

    image_url1 = random.choice(examples_list)
    await query.message.reply_photo(photo=image_url1, caption="Here's a example!  Make sure picture should mainly "
                                                              "contain leaf of the plant")

    await query.answer()
    keyboard = [
        [
            InlineKeyboardButton("Tomato", callback_data=str(tomato)),
            InlineKeyboardButton("Potato", callback_data=str(potato)),
            InlineKeyboardButton("Pepper", callback_data=str(pepper)),
        ]
    ]
    # should show examples
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.reply_text(
        text="Choose the one of these categories: ", reply_markup=reply_markup
    )
    return START_ROUTES


async def predict_photo(update, context):
    file = await context.bot.get_file(update.message.photo[-1].file_id)
    # # download the file as a byte array
    file_bytes = await file.download_as_bytearray()
    # # create a BytesIO object from the byte array
    f = BytesIO(file_bytes)
    # # read the BytesIO object as a numpy array
    file_np = np.frombuffer(f.read(), np.uint8)
    # # decode the numpy array as an image
    img = cv2.imdecode(file_np, cv2.IMREAD_COLOR)
    # # convert the image to BGR format
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # # learn.dls.to(device='cuda')
    # # learn.model.to(device='cuda')
    # # labels = learn.dls.vocab
    # # make a prediction using the model
    cls, v, preds = learn.predict(img)
    val, idx = preds.topk(3)
    classes = learn.dls.vocab[idx]
    if classes[0].split('_')[0].lower() != TYPES:
        text = (f"You choose category {TYPES} but prediction giving {classes[0]}"
                f" as best result, recheck if the image is chosen correctly. "
                f"if chosen image category is correct but we still get different "
                f"result our model can't handle this image. Sorry for inconvenience!\n")
    else:
        text = ""

    for i in range(3):
        text += f"{classes[i]}: {str(int(val[i] * 100))}%"
        text += "\n"
    # # send the prediction as a reply to the user
    # await update.message.photo(img)
    await update.message.reply_text(text=text)
    keyboard = [
        [
            InlineKeyboardButton("Tomato", callback_data=str(tomato)),
            InlineKeyboardButton("Potato", callback_data=str(potato)),
            InlineKeyboardButton("Pepper", callback_data=str(pepper)),
            InlineKeyboardButton("End process", callback_data=str(end)),
        ]
    ]
    # should show examples
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        text="Choose the one of these categories: ", reply_markup=reply_markup
    )
    return START_ROUTES


async def tomato(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global TYPES
    TYPES = 'tomato'
    query = update.callback_query
    await query.message.reply_text('Send the picture, tomato')


async def potato(update, context):
    global TYPES
    TYPES = 'potato'
    query = update.callback_query
    await query.message.reply_text('Send the picture, potato')


async def pepper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global TYPES
    TYPES = 'pepper'
    query = update.callback_query
    await query.message.reply_text('Send the picture, pepper')


async def contact_in(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        text="Please contact us: +998712235566"
    )
    return START_ROUTES


async def contact(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.message.reply_text(
        text="Please contact us: +998712235566"
    )
    return START_ROUTES


async def end(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    # await query.message.reply_text()
    await query.edit_message_text(text="See you next time!")
    return ConversationHandler.END


def main() -> None:
    application = Application.builder().token("6104742729:AAFIveF-9nl5khNyRkEVi7jn-tKMq42QNhI").build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            START_ROUTES: [
                CallbackQueryHandler(category, pattern="^" + str(category) + "$"),
                CallbackQueryHandler(contact, pattern="^" + str(contact) + "$"),
                CallbackQueryHandler(tomato, pattern="^" + str(tomato) + "$"),
                CallbackQueryHandler(potato, pattern="^" + str(potato) + "$"),
                CallbackQueryHandler(pepper, pattern="^" + str(pepper) + "$"),
                CallbackQueryHandler(end, pattern="^" + str(end) + "$"),
                CommandHandler("contact", contact_in),
                CommandHandler("category", category),
                MessageHandler(filters.PHOTO, predict_photo)
            ],
            END_ROUTES: [
                CallbackQueryHandler(start_over, pattern="^" + str(start_over) + "$"),
                CallbackQueryHandler(end, pattern="^" + str(end) + "$"),
            ],
        },
        fallbacks=[CommandHandler("start", start)],
    )
    application.add_handler(conv_handler)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
