# thesisProjectWorkspace
This is a workspace for my current thesis

# Installation:

```
pip install
	flask
	flask+assistant
	nltk
```

# Server Configuration

Go to [ngrok](https://ngrok.com/download) and download the program relative to you operating system. Place the exe downloaded into this file and in the folder type

```
ngrok http 5000
```

This will crate a remote sserver which is mapped to your local host. Copy the https url provided and leave the program running.

# Dialogueflow Configuration

Go to [Dialogflow](https://dialogflow.com) and with a google account, create an agent. Go to the fufilment tab and enable a webhook and put in the https dialogue from ngrok.

Go to general and copy the client and dev tokens. The go to your environmental variables and create a client token 'CLIENT_ACCESS_TOKEN' and paste the client token into its value. Do the same with the dev token under 'DEV_ACCESS_TOKEN'.

You will now need to create two intents. One called 'greeting' and another called 'giveName'. Both intents should have the webhook enabled under the fufilment section. For greeting, add a few training phrases for various greetings and for name, add a few phrases in the vein of 'My name is x'

# Running the program

If all this is done correctly, execute webhook.py via

```
python webhook.py
```

and use the 'Try it out section of dialogue flow. It should ask for you name in which it will either refer to you as 'Mr' or 'Mrs'
