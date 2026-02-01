import { Hono } from 'hono'
import { basicAuth } from 'hono/basic-auth'
import { zValidator } from '@hono/zod-validator'
import * as z from 'zod'
import ollama from 'ollama'
import { Matrix } from 'ml-matrix'
import process from 'node:process';

const themeColor = '\u001b[36;44m'
const app = new Hono();

//PRINTING
function wrapAndIndent(text, width = 60, indent = '  ') {
  const words = text.split(' ');
  const lines = [];
  let currentLine = '';

  for (const word of words) {
    if ((currentLine + word).length > width) {
      lines.push(currentLine.trim());
      currentLine = word + ' ';
    } else {
      currentLine += word + ' ';
    }
  }
  lines.push(currentLine.trim());
  return lines.map(line => indent + line).join('\n');
}

const bold = '\u001b[1m'
const unbold = '\u001b[22m'
const headerColor = '\u001b[37;44m'
const regularColor = '\u001b[37;40m'

process.on('SIGINT', function() {
    console.log("\nCaught interrupt signal");
    console.log(regularColor+"\nExiting.\n")
    process.exit();
});

function printMessage(messageObj){
	console.log(headerColor+bold +'\n'+messageObj.role+':'+unbold+themeColor)
	console.log(wrapAndIndent(messageObj.content))
}

//MESSAGEHANDLING
let messages = []

function addUserMessage(message){
	const userMsgObj = { role: 'user', content: message}
	printMessage(userMsgObj)
	messages = [...messages, userMsgObj] 
}

function addMessage(messageObj){
	printMessage(messageObj)
	messages = [...messages, messageObj]
}

//RAG
// create a vectorstore class
class VectorStore {
    constructor() {
        this.vectors = [];
        this.metadata = [];
    }
    
    add(vector, meta) {
        this.vectors.push(vector);
        this.metadata.push(meta);
    }
    
    // Cosine similarity
    cosineSim(a, b) {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }
    
    search(query, topK = 5) {
        const scores = this.vectors.map((vec, i) => ({
            score: this.cosineSim(query, vec),
            meta: this.metadata[i]
        }));
        return scores.sort((a, b) => b.score - a.score).slice(0, topK);
    }
}

//embeddings-getter
async function getEmbeddings(query){
	const response = await ollama.embed({
		model: 'mxbai-embed-large',
		input: query,
	})
	return response.embeddings
}

//instantiate a vectorstore
class RagMem extends VectorStore{
	constructor(){
		super()
	}

	async addMessage(messageObj, metadata = {}){
		const currentDate = Date.now().toString()
		const embeddings = await getEmbeddings(messageObj.content)
		metadata = Object.assign({}, metadata, {role: messageObj.role, date_received: currentDate, message: messageObj.content})
		super.add(embeddings[0], metadata)
	}

	async queryRagWithMessage(messageObj, topK=5){
		const embeddings = await getEmbeddings(messageObj.content)
		const results = super.search(embeddings[0], topK)
		return results.map(r => ({
			role: r.meta.role,
			content: r.meta.message,
			date_received: r.meta.date_received,
			score: r.score
	        }))
	}

}

const rag = new RagMem()

//ROUTING
app.get('/', zValidator(
    'query',
    z.object({
      chatstr: z.string(),
    })
  ),
  basicAuth({
	username: 'witch',
	password: 'supersecret',
	}),
  async (c) => {
  const { chatstr } = c.req.valid('query')
  addUserMessage(chatstr)
  rag.addMessage(messages.at(-1))
  const ragResults = await rag.queryRagWithMessage(messages.at(-1))
  const chatresponse = await ollama.chat({
  model: 'qwen2.5:1.5b',
  messages: ragResults,
	})
  const returnedResponse = chatresponse.message
  addMessage(returnedResponse)
  return new Response(returnedResponse.content)
})

export default {
	port: 3000,
	fetch: app.fetch,
}
