import { pipeline } from '@xenova/transformers';

const dp=async()=>{
    // Create image-to-text pipeline
const captioner = await pipeline('image-to-text', 'Xenova/trocr-base-handwritten');

// Perform optical character recognition
const image = '1_p.jpg';
const output = await captioner(image);
// [{ generated_text: 'Mr. Brown commented icily.' }]

console.log(output)


}

dp()

