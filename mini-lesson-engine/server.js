import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import dotenv from 'dotenv';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { chromium } from 'playwright';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '5mb' }));
app.use(express.static(join(__dirname, 'public')));

app.get('/',       (_req, res) => res.sendFile(join(__dirname, 'public', 'index.html')));
app.get('/health', (_req, res) => res.json({ ok: true }));

// ── string helpers ────────────────────────────────────────────────────────────

function escapeHTML(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function extractJSON(text) {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced) return JSON.parse(fenced[1].trim());
  const start = text.indexOf('{');
  const end   = text.lastIndexOf('}');
  if (start !== -1 && end > start) return JSON.parse(text.slice(start, end + 1));
  throw new Error('No JSON found in AI response');
}

function ensureArray(val, n = 8) {
  const arr = Array.isArray(val) ? val : [];
  const out = arr.slice(0, n).map(String);
  while (out.length < n) out.push('…');
  return out;
}

function buildNoSpaces(p1, p2) {
  const raw = (p1 + ' ' + p2).replace(/ /g, '');
  return Array.from(
    { length: Math.ceil(raw.length / 60) },
    (_, i) => raw.slice(i * 60, (i + 1) * 60)
  ).join('\n');
}

function buildPhraseRows(lefts, rights) {
  const LETTERS = 'ABCDEFGH';
  const n = 8;
  const L = ensureArray(lefts,  n);
  const R = ensureArray(rights, n);
  const displayOrder = Array.from({ length: n }, (_, i) => (i + 3) % n);

  const rows = L.map((left, i) => {
    const shownRightIdx = displayOrder[i];
    return `<tr>
      <td class="pn">${i + 1}.</td>
      <td class="pl">${escapeHTML(left)}</td>
      <td class="pg"></td>
      <td class="pk">${LETTERS[i]}.</td>
      <td class="pr">${escapeHTML(R[shownRightIdx])}</td>
    </tr>`;
  }).join('\n');

  const answerKey = L.map((_, i) => {
    const displayPos = displayOrder.indexOf(i);
    return `${i + 1}=${LETTERS[displayPos]}`;
  });

  return { rows, answerKey };
}

// ── themes ────────────────────────────────────────────────────────────────────

const THEMES = {
  classic: {
    label: 'Classic',
    css: `
  body { font-family: Arial, Helvetica, sans-serif; font-size: 10.5pt; color: #111; line-height: 1.55; }
  .hdr { border-bottom: 2px solid #111; }
  .hdr h1 { font-size: 16pt; font-weight: 700; }
  .badge { background: #111; color: #fff; padding: 1pt 9pt; font-size: 8pt; letter-spacing: 1.2px; text-transform: uppercase; border-radius: 2px; }
  h2 { font-size: 10.5pt; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; border-bottom: 1px solid #111; padding-bottom: 3pt; }
  h3 { font-size: 10pt; font-weight: 700; }
  .instruct { font-size: 9.5pt; color: #444; font-style: italic; }
  .para-lbl { color: #555; }
  .pk { color: #666; }
  .mono { background: #f8f8f8; border: 1px solid #ccc; font-size: 9.5pt; line-height: 1.8; padding: 9pt 10pt; }
  .wp { border: 1.5px solid #111; padding: 9pt 11pt 5pt; }
  .src { font-size: 7.5pt; color: #666; }
  .ak-item { font-size: 10pt; }
  .ak-pm-row { font-size: 10pt; }
`,
  },

  editorial: {
    label: 'Editorial',
    css: `
  body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; color: #1a1a2e; line-height: 1.6; }
  .hdr { border-bottom: 3px double #1a1a2e; }
  .hdr h1 { font-size: 18pt; font-weight: 700; font-style: italic; letter-spacing: -0.3px; }
  .badge { background: #1a1a2e; color: #f5e6c8; padding: 1pt 9pt; font-size: 7.5pt; letter-spacing: 1.5px; text-transform: uppercase; font-family: Arial, sans-serif; }
  h2 { font-size: 9pt; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; border-bottom: 0.5px solid #999; padding-bottom: 3pt; font-family: Arial, sans-serif; color: #555; }
  h3 { font-size: 10pt; font-weight: 700; font-style: italic; font-family: Georgia, serif; }
  .instruct { font-size: 9pt; color: #555; font-style: italic; }
  .para-lbl { font-family: Arial, sans-serif; font-size: 8pt; text-transform: uppercase; letter-spacing: 0.8px; color: #888; }
  .pk { color: #888; }
  table.pm { font-size: 10pt; }
  .mono { background: #fafaf7; border: 0.5px solid #bbb; border-left: 3px solid #1a1a2e; font-size: 9pt; line-height: 1.9; padding: 9pt 12pt; }
  .wp { border: 0.5px solid #999; border-top: 2px solid #1a1a2e; padding: 9pt 11pt 5pt; }
  .src { font-size: 7.5pt; color: #888; font-style: italic; }
  .ak-item { font-size: 10pt; font-family: Arial, sans-serif; }
  .ak-pm-row { font-size: 10pt; font-family: Arial, sans-serif; }
`,
  },

  compact: {
    label: 'Compact',
    css: `
  body { font-family: Arial, Helvetica, sans-serif; font-size: 9.5pt; color: #111; line-height: 1.45; }
  .hdr { border-bottom: 1.5px solid #111; padding-bottom: 6pt; margin-bottom: 10pt; }
  .hdr h1 { font-size: 13pt; font-weight: 700; }
  .badge { background: #111; color: #fff; padding: 0pt 7pt; font-size: 7.5pt; letter-spacing: 1px; text-transform: uppercase; border-radius: 1px; }
  section { margin-top: 11pt; }
  h2 { font-size: 9pt; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 0.75px solid #111; padding-bottom: 2pt; margin-bottom: 6pt; }
  h3 { font-size: 9pt; font-weight: 700; margin: 6pt 0 4pt; }
  .instruct { font-size: 8.5pt; color: #444; font-style: italic; margin-bottom: 5pt; }
  .para-lbl { font-size: 8pt; color: #666; margin-bottom: 2pt; }
  .reading-p { margin-bottom: 7pt; }
  table.pm { font-size: 9pt; }
  table.pm td { padding: 1pt 2pt; }
  .gap-text { line-height: 1.85; font-size: 9.5pt; }
  .mono { background: #f5f5f5; border: 0.75px solid #ccc; font-size: 8.5pt; line-height: 1.6; padding: 6pt 8pt; }
  .wp { border: 1px solid #111; padding: 7pt 9pt 4pt; }
  .wp-q { margin-bottom: 14pt; }
  .wl { height: 18pt; }
  .src { font-size: 7pt; color: #666; }
  .ak-item { font-size: 9pt; }
  .ak-pm-row { font-size: 9pt; }
  .ak-gf { grid-template-columns: repeat(6, 1fr); }
`,
  },

  modern: {
    label: 'Modern',
    css: `
  body { font-family: -apple-system, 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; color: #0f172a; line-height: 1.55; }
  .hdr { border-bottom: none; background: #0f172a; color: #fff; padding: 12pt 14pt; margin: -18mm -18mm 14pt; }
  .hdr h1 { font-size: 15pt; font-weight: 700; color: #fff; }
  .badge { background: #3b82f6; color: #fff; padding: 1pt 9pt; font-size: 8pt; letter-spacing: 0.8px; text-transform: uppercase; border-radius: 99px; }
  .src { font-size: 7.5pt; color: #94a3b8; margin-top: 5pt; }
  h2 { font-size: 9.5pt; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #3b82f6; border-bottom: 1px solid #e2e8f0; padding-bottom: 3pt; }
  h3 { font-size: 10pt; font-weight: 600; color: #334155; }
  .instruct { font-size: 9.5pt; color: #64748b; font-style: italic; }
  .para-lbl { font-size: 8pt; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #3b82f6; }
  .reading-p { background: #f8fafc; border-left: 3px solid #3b82f6; padding: 6pt 10pt; margin-bottom: 8pt; border-radius: 0 4px 4px 0; }
  .pk { color: #94a3b8; }
  table.pm { font-size: 10pt; }
  .mono { background: #f1f5f9; border: none; border-left: 3px solid #3b82f6; font-size: 9.5pt; line-height: 1.8; padding: 9pt 12pt; }
  .wp { border: none; background: #eff6ff; border-left: 4px solid #3b82f6; padding: 10pt 12pt 6pt; }
  .wl { border-bottom: 1px solid #cbd5e1; height: 21pt; }
  .ak { background: #f8fafc; padding: 14pt; margin: 0 -18mm; padding-left: 18mm; padding-right: 18mm; }
  .ak-item { font-size: 10pt; }
  .ak-pm-row { font-size: 10pt; }
`,
  },
};

// ── HTML template ─────────────────────────────────────────────────────────────

function buildHTML(headline, sources, data, style = 'classic') {
  const { reading, phrase_match, gap_fill, no_spaces, writing_prompt } = data;

  const p1Phrase = buildPhraseRows(phrase_match.p1_left,  phrase_match.p1_right);
  const p2Phrase = buildPhraseRows(phrase_match.p2_left,  phrase_match.p2_right);

  const gapText = escapeHTML(String(gap_fill.text_with_blanks ?? ''))
    .replace(/\((\d{1,2})\)_{2,}/g,
      '<span class="blank">($1)<span class="ul">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span></span>');

  const gapAnswers = ensureArray(gap_fill.answers, 12);

  const sourcesBlock = Array.isArray(sources) && sources.filter(Boolean).length
    ? `<p class="src">Sources: ${sources.filter(Boolean).map(escapeHTML).join(' &bull; ')}</p>`
    : '';

  const writingLines = Array(6).fill('<div class="wl"></div>').join('\n      ');
  const themeCss = (THEMES[style] ?? THEMES.classic).css;

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>${escapeHTML(headline)} – ESL Level 0</title>
<style>
  @page { size: A4; margin: 18mm 18mm 20mm; }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  /* structural */
  .hdr { padding-bottom: 9pt; margin-bottom: 14pt; }
  .badge { display: inline-block; }
  section { margin-top: 16pt; }
  h2 { margin-bottom: 9pt; }
  h3 { margin: 9pt 0 5pt; }
  .instruct { margin-bottom: 7pt; }
  .para-lbl { font-weight: 700; font-size: 9pt; margin-bottom: 3pt; }
  .reading-p { margin-bottom: 10pt; }
  table.pm { width: 100%; border-collapse: collapse; margin-bottom: 4pt; }
  table.pm td { padding: 2pt 3pt; vertical-align: top; }
  .pn { width: 18pt; font-weight: 700; white-space: nowrap; }
  .pl { width: 42%; }
  .pg { width: 12pt; }
  .pk { width: 18pt; font-weight: 700; }
  .gap-text { line-height: 2.1; font-size: 10.5pt; }
  .ul { text-decoration: underline; }
  .mono { font-family: 'Courier New', Courier, monospace; white-space: pre; }
  .wp { padding: 9pt 11pt 5pt; }
  .wp-q { font-weight: 700; margin-bottom: 18pt; }
  .wl { border-bottom: 1px solid #bbb; height: 21pt; }
  .ak { page-break-before: always; }
  .ak-pm { margin-top: 8pt; }
  .ak-pm-row { margin-bottom: 4pt; }
  .ak-gf { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3pt; margin-top: 7pt; }
  @media print { .ak { page-break-before: always; } }

  /* theme */
  ${themeCss}
</style>
</head>
<body>

<div class="hdr">
  <h1>${escapeHTML(headline)}</h1>
  <span class="badge">Level 0 &ndash; Pre-A1/A1</span>
  ${sourcesBlock}
</div>

<section>
  <h2>Reading</h2>
  <div class="reading-p">
    <div class="para-lbl">Paragraph 1</div>
    <p>${escapeHTML(reading.p1)}</p>
  </div>
  <div class="reading-p">
    <div class="para-lbl">Paragraph 2</div>
    <p>${escapeHTML(reading.p2)}</p>
  </div>
</section>

<section>
  <h2>Exercise 1 &ndash; Phrase Match</h2>
  <p class="instruct">Match the beginnings (1&ndash;8) with the correct endings (A&ndash;H).</p>
  <h3>Paragraph 1</h3>
  <table class="pm"><tbody>${p1Phrase.rows}</tbody></table>
  <h3>Paragraph 2</h3>
  <table class="pm"><tbody>${p2Phrase.rows}</tbody></table>
</section>

<section>
  <h2>Exercise 2 &ndash; Gap Fill</h2>
  <p class="instruct">Fill in the gaps (1)&ndash;(12) with words from the reading.</p>
  <div class="gap-text">${gapText}</div>
</section>

<section>
  <h2>Exercise 3 &ndash; Put a Slash Where the Spaces Are</h2>
  <p class="instruct">All spaces have been removed. Write a slash (/) where each space should go.</p>
  <div class="mono">${escapeHTML(no_spaces.text_wrapped_60)}</div>
</section>

<section>
  <h2>Writing Prompt</h2>
  <div class="wp">
    <p class="wp-q">${escapeHTML(writing_prompt)}</p>
    ${writingLines}
  </div>
</section>

<div class="ak">
  <h2>Answer Key</h2>

  <h3>Exercise 1 &ndash; Phrase Match</h3>
  <div class="ak-pm">
    <div class="ak-pm-row"><strong>Paragraph 1:</strong> ${p1Phrase.answerKey.join(' &bull; ')}</div>
    <div class="ak-pm-row"><strong>Paragraph 2:</strong> ${p2Phrase.answerKey.join(' &bull; ')}</div>
  </div>

  <h3 style="margin-top:12pt">Exercise 2 &ndash; Gap Fill</h3>
  <div class="ak-gf">
    ${gapAnswers.map((a, i) => `<div class="ak-item"><strong>(${i + 1})</strong> ${escapeHTML(a)}</div>`).join('\n    ')}
  </div>
</div>

</body>
</html>`;
}

// ── AI: call Gemini ───────────────────────────────────────────────────────────

async function generateWithAI(headline, article) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'YOUR_KEY_HERE') throw new Error('NO_API_KEY');

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: 'gemini-2.0-flash',
    generationConfig: {
      temperature: 0.7,
      responseMimeType: 'application/json',
    },
  });

  const prompt = `You are an expert ESL teacher creating a CEFR Level 0 (Pre-A1 / A1) lesson.

HEADLINE: ${headline}

ARTICLE (use this as source material):
${article.slice(0, 4000)}

Return ONLY a valid JSON object with this exact structure:

{
  "reading": {
    "p1": "3-5 very simple sentences about the main news. Use: subject + verb + object. Present tense. Common words (top 2000). No idioms.",
    "p2": "3-5 very simple sentences. Start with Also, or However, or But. Continue the story simply."
  },
  "phrase_match": {
    "p1_left":  ["3-5 word start of phrase 1 from p1", "…", "…", "…", "…", "…", "…", "…"],
    "p1_right": ["ending of phrase 1 matching p1_left[0]", "…", "…", "…", "…", "…", "…", "…"],
    "p2_left":  ["3-5 word start of phrase 1 from p2", "…", "…", "…", "…", "…", "…", "…"],
    "p2_right": ["ending of phrase 1 matching p2_left[0]", "…", "…", "…", "…", "…", "…", "…"]
  },
  "gap_fill": {
    "text_with_blanks": "Full text of p1 and p2 combined. Replace exactly 12 content words with (1)____ through (12)____. Keep all other words unchanged.",
    "answers": ["word1","word2","word3","word4","word5","word6","word7","word8","word9","word10","word11","word12"]
  },
  "writing_prompt": "One simple why/why not question about the topic. E.g. 'Why do you think … is important? Write 3-5 sentences.'"
}

CRITICAL RULES:
- reading: CEFR A1 max. Very short sentences. Only common words.
- phrase_match: p1_left[i] + " " + p1_right[i] must form a phrase that appears in p1. EXACTLY 8 items per array.
- phrase_match: p2_left[i] + " " + p2_right[i] must form a phrase that appears in p2. EXACTLY 8 items per array.
- gap_fill answers: EXACTLY 12 items in the same order as (1)–(12) in text_with_blanks.
- Output only the JSON, nothing else.`;

  const result = await model.generateContent(prompt);
  const text   = result.response.text();

  try {
    return JSON.parse(text);
  } catch {
    return extractJSON(text);
  }
}

// ── mock data (no API key) ────────────────────────────────────────────────────

function getMockData(headline) {
  const topic = headline.slice(0, 30);
  const p1 = `There is important news about ${topic}. Many people talk about this story. It happened recently. Everyone wants to know more.`;
  const p2 = `Also, experts say this is a big change. However, not everyone agrees. But the news is very clear. This affects many people around the world.`;
  return {
    reading: { p1, p2 },
    phrase_match: {
      p1_left:  ['There is important', 'Many people', 'It happened', 'Everyone wants', 'The story is', 'People read', 'This news', 'We can all'],
      p1_right: [`news about ${topic.slice(0,12)}`, 'talk about this story', 'recently', 'to know more', 'very big', 'about it', 'is true', 'learn from it'],
      p2_left:  ['Also experts say', 'However not everyone', 'But the news', 'This affects', 'People around', 'It is a', 'We should', 'The change'],
      p2_right: ['this is a big change', 'agrees', 'is very clear', 'many people', 'the world', 'big story', 'pay attention', 'is important'],
    },
    gap_fill: {
      text_with_blanks: `There is important (1)____ about ${topic.slice(0,10)}. Many (2)____ talk about this story. It (3)____ recently. Everyone wants to (4)____ more. Also, experts say this is a big (5)____. However, not everyone (6)____. But the news is very (7)____. This (8)____ many people around the world. We can all (9)____ from this. It is a (10)____ story. People need to (11)____ attention. The (12)____ is important.`,
      answers: ['news','people','happened','know','change','agrees','clear','affects','learn','big','pay','story'],
    },
    writing_prompt: `Why do you think the news about "${topic}" is important? Write 3–5 sentences.`,
  };
}

// ── endpoints ─────────────────────────────────────────────────────────────────

app.post('/api/generate-level0', async (req, res) => {
  try {
    const { headline, article, sources, style } = req.body ?? {};
    if (!headline || !article) {
      return res.status(400).json({ error: 'headline and article are required' });
    }

    let aiData;
    try {
      aiData = await generateWithAI(headline, article);
    } catch (err) {
      if (err.message === 'NO_API_KEY') {
        console.warn('[mock] GEMINI_API_KEY not set – returning mock lesson');
        aiData = getMockData(headline);
      } else {
        throw err;
      }
    }

    const noSpacesText = buildNoSpaces(
      aiData.reading?.p1 ?? '',
      aiData.reading?.p2 ?? ''
    );
    const fullData = {
      reading:        aiData.reading        ?? { p1: '', p2: '' },
      phrase_match:   aiData.phrase_match   ?? { p1_left:[], p1_right:[], p2_left:[], p2_right:[] },
      gap_fill:       aiData.gap_fill       ?? { text_with_blanks: '', answers: [] },
      no_spaces:      { text_wrapped_60: noSpacesText },
      writing_prompt: aiData.writing_prompt ?? '',
    };

    const html = buildHTML(headline, sources, fullData, style);

    res.json({ ...fullData, html });
  } catch (err) {
    console.error('[generate-level0]', err);
    res.status(500).json({ error: err.message || 'Internal server error' });
  }
});

// Re-render with a different style (no AI call — instant)
app.post('/api/render', (req, res) => {
  try {
    const { headline, sources, data, style } = req.body ?? {};
    if (!headline || !data) {
      return res.status(400).json({ error: 'headline and data are required' });
    }
    const html = buildHTML(headline, sources, data, style);
    res.json({ html });
  } catch (err) {
    console.error('[render]', err);
    res.status(500).json({ error: err.message || 'Render failed' });
  }
});

// Expose available theme names to the frontend
app.get('/api/themes', (_req, res) => {
  res.json(Object.entries(THEMES).map(([id, t]) => ({ id, label: t.label })));
});

app.post('/api/pdf', async (req, res) => {
  let browser;
  try {
    const { html } = req.body ?? {};
    if (!html) return res.status(400).json({ error: 'html is required' });

    const launchOpts = {};
    const sysBin = process.env.CHROMIUM_EXECUTABLE_PATH
      || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
    try {
      const { statSync } = await import('fs');
      statSync(sysBin);
      launchOpts.executablePath = sysBin;
    } catch { /* use Playwright's default if the path doesn't exist */ }

    browser = await chromium.launch(launchOpts);
    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: 'networkidle' });
    const pdf = await page.pdf({ format: 'A4', printBackground: true });
    await browser.close();
    browser = null;

    res.set({
      'Content-Type':        'application/pdf',
      'Content-Disposition': 'attachment; filename="lesson-level0.pdf"',
      'Content-Length':      pdf.length,
    });
    res.send(pdf);
  } catch (err) {
    if (browser) await browser.close().catch(() => {});
    console.error('[pdf]', err);
    res.status(500).json({ error: err.message || 'PDF generation failed' });
  }
});

// ── start ─────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`ESL Lesson Engine → http://localhost:${PORT}`);
  if (!process.env.GEMINI_API_KEY || process.env.GEMINI_API_KEY === 'YOUR_KEY_HERE') {
    console.warn('  ⚠  GEMINI_API_KEY not set — mock mode active');
  }
});
