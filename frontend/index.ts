import { generateText } from 'ai';
import { deepseek } from '@ai-sdk/deepseek';

const { text } = await generateText({
  model: deepseek('deepseek-flash'),
  prompt: 'Invent a new holiday and describe its traditions.',
});

console.log(text);
