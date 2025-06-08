import discord
from discord.ext import commands
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import os

IMG_HEIGHT = 224
IMG_WIDTH = 224

class_names = ["dragon_ball_z", "my_hero_academia","naruto"]

TOKEN = os.getenv("TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    global model
    model = tf.keras.models.load_model("best_anime_classifier.h5")
    print(f"[bot.py] Conectado como {bot.user}")

@bot.command(name="classificar")
async def classificar(ctx):
    if not ctx.message.attachments:
        await ctx.send("📌 Envie uma imagem junto ao comando para classificação.")
        return

    # Pega o primeiro anexo
    attachment = ctx.message.attachments[0]
    img_bytes = await attachment.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Redimensiona para as dimensões utilizadas no treinamento
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Aplica preprocess_input do EfficientNet
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Faz a predição
    preds = model.predict(img_array)
    idx = np.argmax(preds, axis=1)[0]
    classe = class_names[idx]
    confianca = float(np.max(preds))

    await ctx.send(f"🔍 Resultado: **{classe}** (confiança: {confianca:.2f})")
    TOKEN = os.getenv("DISCORD_TOKEN")

bot.run(TOKEN)
