# -- coding: utf-8 --
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
# from telegram import KeyboardButton, KeyboardButtonPollType
import apiai
import json
import os
import dialogflow
import re
import logging

import string
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import annoy
from gensim.models import KeyedVectors  # Word2Vec, FastText,
# import pickle
import numpy as np
# from tqdm import tqdm
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from linecache import getline
# import compress_fasttext
from gensim.models.fasttext import FastTextKeyedVectors
from datetime import datetime

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "project-nlp-cw-bot-khaj-4c902d5b3fdc.json"

DIALOGFLOW_PROJECT_ID = 'project-nlp-cw-bot-khaj'
DIALOGFLOW_LANGUAGE_CODE = 'ru'
SESSION_ID = 'glvv_nlp_cw'

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


def start(update, context):
    """Send a message when the command /start is issued."""
    txt = 'Здравствуйте! Бот найдёт ответы в базе mail.ru. Признаком вопроса является знак вопроса :)'
    logger.info(f"Start command: {txt}")
    update.message.reply_text(txt)


def help_command(update, context):
    """Send a message when the command /help is issued."""
    txt = "Задайте вопрос и бот найдёт самые подходящие вопросы-ответы в базе Ответов Mail.ru. \
        Признаком вопроса является знак вопроса :)"
    logger.info(f"Help command: {txt}")
    update.message.reply_text(txt)
    # button = [[KeyboardButton("Press me!", request_poll=KeyboardButtonPollType())]]


w2v_width = 100
ft_width = 300
flag_w2v = False
flag_ft = False
flag_w2v_index = False
flag_ft_index = False
w2v_index = annoy.AnnoyIndex(w2v_width, 'angular')
ft_index = annoy.AnnoyIndex(ft_width, 'angular')

w2v_file = Path("word2vec_wv.bin")
if w2v_file.is_file():
    # modelW2V = Word2Vec.load(w2v_file)
    model_w2v_wv = KeyedVectors.load_word2vec_format(w2v_file, binary=True)
    flag_w2v = True

ft_file = Path("ft_freqprune_400K_100K_pq_300.bin")
if ft_file.is_file():
    model_ft_wv = FastTextKeyedVectors.load("ft_freqprune_400K_100K_pq_300.bin")
    flag_ft = True

prep_answ = "prepared_answers.txt"
# if index_file.is_file():
#    with open('index_map.pickle', 'rb') as f:
#         index_map = pickle.load(f)
#    flag_index_map = True

w2v_index_file = Path("w2v_index.ann")
if w2v_index_file.is_file():
    w2v_index.load("w2v_index.ann")
    flag_w2v_index = True

ft_index_file = Path("ft_index.ann")
if ft_index_file.is_file():
    ft_index.load("ft_index.ann")
    flag_ft_index = True

morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation)


def preprocess_txt(line):
    spls = "".join(i for i in line.strip() if i not in exclude).split()
    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = [i for i in spls if i not in sw and i != ""]
    return spls


def get_response(question, index, model, ans_num: int, width: int) -> List:
    question = preprocess_txt(question)
    vector = np.zeros(width)
    norm = 0
    for word in question:
        if word in model:
            vector += model[word]
            norm += 1
    if norm > 0:
        vector = vector / norm
    return index.get_nns_by_vector(vector, ans_num)


def echo(update, context):
    """Echo the user message."""
    # update.message.reply_text(update.message.text)
    # update.message.reply_text('Ваше сообщение принял: ' + update.message.text.lower())
    # update.message.reply_text(f"Поиск признака вопроса: {re.findall(r'[?]', update.message.text)}")
    logger.info(f"Request: {update.message.text}")
    question = re.search(r'[?]', update.message.text)
    if question is not None:
        if flag_w2v and flag_w2v_index:
            line_numbers = get_response(update.message.text, w2v_index, model_w2v_wv, 2, w2v_width)
            spls = [getline(prep_answ, i+1).split("\t") for i in line_numbers]

            txt = "Word2vec 100:"
            logger.info(f"\t{txt}")
            update.message.reply_text(f"<b>{txt}</b>", parse_mode=telegram.ParseMode.HTML)

            for i in range(2):
                # print(f"Вопрос: {spl[0]}\nОтвет: {spl[1]}")
                answer = re.sub(r'<br>', '\n', spls[i][1])
                answer = re.sub(r'<[^<]*>', '', answer)
                str_out = f"*Вопрос:*\n{spls[i][0]}\n\n*Ответ:*\n{answer}"
                logger.info(f"\t{str_out[:4096]}")
                update.message.reply_text(str_out[:4096], parse_mode=telegram.ParseMode.MARKDOWN)
        if flag_ft and flag_ft_index:
            line_numbers = get_response(update.message.text, ft_index, model_ft_wv, 2, ft_width)
            spls = [getline(prep_answ, i+1).split("\t") for i in line_numbers]

            txt = "FastText quant 300:"
            logger.info(f"\t{txt}")
            update.message.reply_text(f"<b>{txt}</b>", parse_mode=telegram.ParseMode.HTML)

            for i in range(2):
                answer = re.sub(r'<br>', '\n', spls[i][1])
                answer = re.sub(r'<[^<]*>', '', answer)
                str_out = f"*Вопрос:*\n{spls[i][0]}\n\n*Ответ:*\n{answer}"
                logger.info(f"\t{str_out[:4096]}")
                update.message.reply_text(str_out[:4096], parse_mode=telegram.ParseMode.MARKDOWN)
    else:
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
        text_input = dialogflow.types.TextInput(text=update.message.text, language_code=DIALOGFLOW_LANGUAGE_CODE)
        query_input = dialogflow.types.QueryInput(text=text_input)

        try:
            response = session_client.detect_intent(session=session, query_input=query_input)
        except InvalidArgument:
            raise

        text = response.query_result.fulfillment_text

        if text:
            logger.info(f"\t{text}")
            update.message.reply_text(text=text)
        else:
            txt = 'Не понял...'
            logger.info(f"\t{txt}")
            update.message.reply_text(text=txt)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater('1394117290:AAEVn7j-CNn-rJ-9eHxxxxxxxxxx', use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()

