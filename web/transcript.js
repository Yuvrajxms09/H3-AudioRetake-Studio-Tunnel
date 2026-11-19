'use strict';

// Original word timing stays immutable; the editable script is a separate draft.
const transcript = {words: [], anchor: null, draft: '', reviewed: null};
const WORD_PAD = .03;
const TRANSCRIPT_DIFF_MAX_CELLS = 4000000;

function sourceTranscriptText() {
 return transcript.words.map(word => String(word.word || '').trim()).filter(Boolean).join(' ');
}

function transcriptWordKey(text) {
 return String(text || '').normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
}

function transcriptTokens(text) {
 return String(text || '').trim().split(/\s+/).filter(Boolean).map(text => ({
  text,
  key: transcriptWordKey(text)
 }));
}

function transcriptLcsMatches(oldTokens, newTokens) {
 const rowsAreOld = oldTokens.length <= newTokens.length;
 const rows = rowsAreOld ? oldTokens : newTokens;
 const columns = rowsAreOld ? newTokens : oldTokens;
 const cells = (rows.length + 1) * (columns.length + 1);
 if (cells > TRANSCRIPT_DIFF_MAX_CELLS) {
  throw Error('This transcript is too long to compare in one pass. Use a shorter source clip, then transcribe and edit that clip.');
 }

 const lcs = Array.from({length: rows.length + 1}, () => new Uint32Array(columns.length + 1));
 for (let i = 1; i <= rows.length; i++) {
  for (let j = 1; j <= columns.length; j++) {
   lcs[i][j] = rows[i - 1].key === columns[j - 1].key
    ? lcs[i - 1][j - 1] + 1
    : Math.max(lcs[i - 1][j], lcs[i][j - 1]);
  }
 }

 const matches = [];
 let i = rows.length;
 let j = columns.length;
 while (i && j) {
  if (rows[i - 1].key === columns[j - 1].key) {
   matches.push(rowsAreOld ? {oldIndex: i - 1, newIndex: j - 1} : {oldIndex: j - 1, newIndex: i - 1});
   i--;
   j--;
  } else if (lcs[i - 1][j] >= lcs[i][j - 1]) {
   i--;
  } else {
   j--;
  }
 }
 return matches.reverse();
}

function transcriptDiffHunks(oldTokens, newTokens) {
 const matches = transcriptLcsMatches(oldTokens, newTokens);
 const hunks = [];
 let oldAt = 0;
 let newAt = 0;
 for (const match of matches) {
  if (match.oldIndex > oldAt || match.newIndex > newAt) {
   hunks.push({oldStart: oldAt, oldEnd: match.oldIndex, newStart: newAt, newEnd: match.newIndex});
  }
  oldAt = match.oldIndex + 1;
  newAt = match.newIndex + 1;
 }
 if (oldAt < oldTokens.length || newAt < newTokens.length) {
  hunks.push({oldStart: oldAt, oldEnd: oldTokens.length, newStart: newAt, newEnd: newTokens.length});
 }
 return hunks;
}

function changeText(tokens, start, end) {
 return tokens.slice(start, end).map(token => token.text).join(' ').trim();
}

function mapTranscriptChanges(draft) {
 if (!transcript.words.length) throw Error('Transcribe the source first so the edited words can be mapped to their original times.');
 const oldTokens = transcript.words.map(word => ({text: String(word.word || '').trim()}));
 oldTokens.forEach(token => { token.key = transcriptWordKey(token.text); });
 const newTokens = transcriptTokens(draft);
 const hunks = transcriptDiffHunks(oldTokens, newTokens);
 const changes = hunks.map(hunk => {
  let oldStart = hunk.oldStart;
  let oldEnd = hunk.oldEnd;
  let newStart = hunk.newStart;
  let newEnd = hunk.newEnd;
  const insertion = oldStart === oldEnd;
  const deletion = newStart === newEnd;
  const kind = insertion ? 'addition' : deletion ? 'deletion' : 'replacement';
  const expandedForContext = insertion || deletion;

  // Joint Retake needs a source interval. For insertions/deletions, include the
  // nearest retained words and ask H3 to speak that short phrase around the edit.
  if (insertion || deletion) {
   oldStart = Math.max(0, oldStart - 1);
   oldEnd = Math.min(oldTokens.length, oldEnd + 1);
   newStart = Math.max(0, newStart - 1);
   newEnd = Math.min(newTokens.length, newEnd + 1);
  }

  const first = transcript.words[oldStart];
  const last = transcript.words[oldEnd - 1];
  const start = first ? Math.max(0, Number(first.start) - WORD_PAD) : NaN;
  const end = last && state.buffer ? Math.min(state.buffer.duration, Number(last.end) + WORD_PAD) : last ? Number(last.end) + WORD_PAD : NaN;
  const selectedSourceText = changeText(oldTokens, oldStart, oldEnd);
  const targetText = changeText(newTokens, newStart, newEnd);
  let mappingIssue = '';

  if (!first || !last || !Number.isFinite(start) || !Number.isFinite(end)) {
   mappingIssue = 'The source words do not have usable timing. Adjust this passage manually.';
  } else if (!targetText) {
   mappingIssue = 'This edit removes all remaining speech in its passage. Joint Retake needs replacement words; enter a short replacement line to continue.';
  }

  return {
   kind,
   oldStart,
   oldEnd,
   newStart,
   newEnd,
   start: Number.isFinite(start) ? start : 0,
   end: Number.isFinite(end) ? end : 0,
   sourceText: selectedSourceText,
   targetText,
   expandedForContext,
   mappingIssue
  };
 });

 // Context expansion can make neighboring edits overlap. Merge those passages
 // so they become one coherent line instead of two competing H3 jobs.
 const merged = [];
 for (const change of changes) {
  const previous = merged[merged.length - 1];
  if (previous && change.start < previous.end) {
   previous.oldStart = Math.min(previous.oldStart, change.oldStart);
   previous.oldEnd = Math.max(previous.oldEnd, change.oldEnd);
   previous.newStart = Math.min(previous.newStart, change.newStart);
   previous.newEnd = Math.max(previous.newEnd, change.newEnd);
   previous.start = Math.min(previous.start, change.start);
   previous.end = Math.max(previous.end, change.end);
   previous.expandedForContext = true;
   previous.kind = 'combined edits';
   previous.sourceText = changeText(oldTokens, previous.oldStart, previous.oldEnd);
   previous.targetText = changeText(newTokens, previous.newStart, previous.newEnd);
   previous.mappingIssue = previous.mappingIssue || change.mappingIssue;
  } else {
   merged.push({...change});
  }
 }
 return merged;
}

function transcriptChangeIssues(changes) {
 const issues = new Map();
 for (let i = 0; i < changes.length; i++) {
  const change = changes[i];
  if (change.mappingIssue) issues.set(i, change.mappingIssue);
  if (!Number.isFinite(change.start) || !Number.isFinite(change.end) || change.start < 0 || change.end <= change.start) {
   issues.set(i, 'Enter a valid start and end time.');
  }
  if (state.buffer && change.end > state.buffer.duration) {
   issues.set(i, 'End time is outside the source duration.');
  }
  const maxSelection = JOINT_MAX_FRAMES / 24 - 2 * JOINT_MARGIN;
  if (change.end - change.start > maxSelection) {
   issues.set(i, 'This changed passage exceeds Joint Retake’s per-pass limit. Split it into shorter edits.');
  }
 }
 for (let i = 0; i < changes.length; i++) {
  for (let j = i + 1; j < changes.length; j++) {
   if (changes[i].start < changes[j].end && changes[i].end > changes[j].start) {
    issues.set(i, issues.get(i) || 'This passage overlaps another changed passage. Combine or adjust the ranges.');
    issues.set(j, issues.get(j) || 'This passage overlaps another changed passage. Combine or adjust the ranges.');
   }
  }
 }
 return issues;
}

function transcriptQueueIssue() {
 if (!transcript.reviewed) return 'Review the edited transcript before queueing retakes.';
 if (transcript.reviewed.draft !== $('transcriptDraft').value) return 'The transcript changed after review. Review the changes again.';
 if (!transcript.reviewed.changes.length) return 'There are no spoken-word changes to queue.';
 if (!state.sourceFile || !state.sourceFile.type.startsWith('video/')) return 'Open a video source before queueing Joint Retakes.';
 const duration = state.buffer?.duration ?? video.duration;
 if (!Number.isFinite(duration) || duration * 24 < JOINT_MIN_FRAMES) return 'Joint Retake needs at least about 5.2 seconds of source video.';
 if (transcriptChangeIssues(transcript.reviewed.changes).size) return 'Fix the timing or unsupported edits shown below before queueing.';
 if (dialogue.turns.length + transcript.reviewed.changes.length > 8) return 'The Studio supports up to eight dialogue turns. Remove turns or queue fewer changed passages.';
 if (dialogue.turns.some(turn => turn.transcriptEdit)) return 'Transcript edits are already queued. Remove those turns before queueing a revised transcript.';

 const lane = laneOf('joint');
 const occupied = dialogue.turns.filter(turn => !tracksOn() || turnLane(turn) === lane);
 for (const change of transcript.reviewed.changes) {
  if (occupied.some(turn => change.start < turn.end - .0001 && change.end > turn.start + .0001)) {
   return 'A changed passage overlaps an existing turn on the same track. Remove or adjust the conflicting turn first.';
  }
 }
 return '';
}

function setTranscriptRange(change) {
 leaveTurn();
 renderTurns();
 state.start = change.start;
 state.end = change.end;
 syncSelection();
 $('prompt').value = change.targetText;
 $('prompt').dispatchEvent(new Event('input'));
}

function renderTranscriptChanges() {
 const list = $('transcriptChangeList');
 const queue = $('queueTranscriptEdits');
 const status = $('transcriptEditStatus');
 list.replaceChildren();
 const reviewed = transcript.reviewed;
 list.hidden = !reviewed || !reviewed.changes.length;
 if (!reviewed) {
  queue.disabled = true;
  status.textContent = transcript.words.length ? 'Edit the script, then review its changed passages.' : 'Transcribe a source to start editing its script.';
  return;
 }
 if (!reviewed.changes.length) {
  queue.disabled = true;
  status.textContent = 'No spoken-word changes found. Capitalization and punctuation-only edits to the same words do not need a speech retake.';
  return;
 }

 const issues = transcriptChangeIssues(reviewed.changes);
 reviewed.changes.forEach((change, index) => {
  const card = document.createElement('article');
  card.className = 'transcript-change' + (issues.has(index) ? ' has-issue' : '');
  const heading = document.createElement('div');
  heading.className = 'transcript-change-heading';
  const title = document.createElement('strong');
  title.textContent = 'Change ' + (index + 1) + ' · ' + fmt(change.start) + '–' + fmt(change.end);
  const select = document.createElement('button');
  select.type = 'button';
  select.className = 'textbutton';
  select.textContent = 'Select passage';
  select.onclick = () => setTranscriptRange(change);
  heading.append(title, select);

  const pair = document.createElement('div');
  pair.className = 'transcript-change-pair';
  const oldLine = document.createElement('p');
  const newLine = document.createElement('p');
  const oldLabel = document.createElement('b');
  const newLabel = document.createElement('b');
  oldLabel.textContent = 'Source words';
  newLabel.textContent = 'Words to speak';
  oldLine.append(oldLabel, document.createTextNode(change.sourceText || '(no source words)'));
  newLine.append(newLabel, document.createTextNode(change.targetText || '(empty)'));
  pair.append(oldLine, newLine);

  const timeControls = document.createElement('div');
  timeControls.className = 'transcript-change-times';
  for (const [field, label] of [['start', 'IN'], ['end', 'OUT']]) {
   const wrapper = document.createElement('label');
   wrapper.textContent = label;
   const input = document.createElement('input');
   input.type = 'number';
   input.step = '0.01';
   input.min = '0';
   input.max = String(state.buffer?.duration || 0);
   input.value = Number(change[field]).toFixed(3);
   input.setAttribute('aria-label', 'Change ' + (index + 1) + ' ' + label.toLowerCase() + ' time in seconds');
   input.onchange = () => {
    const value = input.valueAsNumber;
    if (Number.isFinite(value)) change[field] = value;
    renderTranscriptChanges();
   };
   wrapper.append(input);
   timeControls.append(wrapper);
  }
  const useSelection = document.createElement('button');
  useSelection.type = 'button';
  useSelection.className = 'textbutton';
  useSelection.textContent = 'Use waveform selection';
  useSelection.onclick = () => {
   readRange();
   if (state.end <= state.start) {
    status.textContent = 'Select a non-empty waveform range first.';
    return;
   }
   change.start = state.start;
   change.end = state.end;
   renderTranscriptChanges();
  };
  timeControls.append(useSelection);
  card.append(heading, pair, timeControls);
  if (change.expandedForContext) {
   const context = document.createElement('p');
   context.className = 'transcript-change-note';
   context.textContent = 'This passage includes nearby words so Joint Retake has a source interval for the insertion or deletion.';
   card.append(context);
  }
  if (issues.has(index)) {
   const note = document.createElement('p');
   note.className = 'transcript-change-error';
   note.textContent = issues.get(index);
   card.append(note);
  }
  list.append(card);
 });

 const issue = transcriptQueueIssue();
 queue.disabled = Boolean(issue);
 queue.textContent = dialogue.turns.some(turn => turn.transcriptEdit)
  ? 'Remove queued transcript turns to make a new set'
  : 'Queue changed passages · 5 options each';
 status.textContent = issue || 'Review each mapped passage. Queueing creates one Joint Retake turn per passage, with five H3 variations for comparison.';
}

function renderTranscriptEditor() {
 const section = $('transcriptEditor');
 section.hidden = !transcript.words.length;
 const editor = $('transcriptDraft');
 if (transcript.words.length && editor.value !== transcript.draft) editor.value = transcript.draft;
 renderTranscriptChanges();
}

function selectWords(from, to) {
 const a = transcript.words[Math.min(from, to)];
 const b = transcript.words[Math.max(from, to)];
 state.start = Math.max(0, a.start - WORD_PAD);
 state.end = Math.min(state.buffer.duration, b.end + WORD_PAD);
 syncSelection();
 $('prompt').value = transcript.words.slice(Math.min(from, to), Math.max(from, to) + 1).map(word => word.word).join(' ');
 $('prompt').dispatchEvent(new Event('input'));
}

function markTranscript() {
 const buttons = $('transcript').children;
 transcript.words.forEach((word, index) => {
  buttons[index]?.classList.toggle('selected', word.start < state.end && word.end > state.start && state.end > state.start);
 });
}

function renderTranscript() {
 const box = $('transcript');
 box.replaceChildren();
 box.hidden = !transcript.words.length;
 transcript.words.forEach((word, index) => {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = word.word;
  button.title = fmt(word.start) + ' – ' + fmt(word.end);
  button.onclick = event => {
   const from = event.shiftKey && transcript.anchor !== null ? transcript.anchor : index;
   if (!event.shiftKey) transcript.anchor = index;
   selectWords(from, index);
  };
  box.append(button);
 });
 markTranscript();
 renderTranscriptEditor();
}

async function speechBuffer() {
 const sampleRate = 16000;
 const offline = new OfflineAudioContext(1, Math.ceil(state.buffer.duration * sampleRate), sampleRate);
 const node = offline.createBufferSource();
 node.buffer = state.buffer;
 node.connect(offline.destination);
 node.start();
 return offline.startRendering();
}

$('transcribe').onclick = async () => {
 if (!state.buffer) return message('Load a source first.');
 const button = $('transcribe');
 button.disabled = true;
 try {
  message('Transcribing the source locally…');
  const name = await uploadStudio(wav(await speechBuffer()), 'transcript_' + crypto.randomUUID() + '.wav');
  const outputs = await queueLipGraph({
   '1': {class_type: 'LoadAudio', inputs: {audio: 'AudioRepairStudio/' + name}},
   '2': {class_type: 'AudioRetakeTranscribe', inputs: {audio: ['1', 0], language: $('language').value.trim()}}
  });
  transcript.words = JSON.parse(outputs['2'].transcript[0]);
  transcript.anchor = null;
  transcript.draft = sourceTranscriptText();
  transcript.reviewed = null;
  renderTranscript();
  checkLine();
  window.dispatchEvent(new Event('studio-project-changed'));
  message('Transcript ready. Click words to select a passage, or edit the full script below.');
 } catch (error) {
  message('Transcription failed: ' + error.message);
 } finally {
  button.disabled = false;
 }
};

function syllables(text) {
 return text.toLowerCase().split(/\s+/).reduce((total, word) => {
  const letters = word.replace(/[^\p{L}]/gu, '');
  return letters ? total + Math.max(1, (letters.replace(/(?<=[^aeiouy])es?$/, '').match(/[aeiouyàâäéèêëîïôöùûüœ]+/g) || []).length) : total;
 }, 0);
}

// Estimate the speaker's pace from timed words, otherwise use an ordinary rate.
function speakingRate() {
 const rates = transcript.words.filter(word => word.end - word.start >= .1)
  .map(word => syllables(word.word) / (word.end - word.start)).sort((a, b) => a - b);
 return rates.length >= 5 ? clamp(rates[Math.floor(rates.length / 2)], 3, 8) : 5.5;
}

function checkLine() {
 const line = $('prompt').value.trim();
 const note = $('lineCheck');
 const issues = [];
 if (line) {
  if (/\.\.\.|…/.test(line)) issues.push('Ellipses often hurt clarity.');
  if (/["“”«»]/.test(line)) issues.push('Remove quotation marks.');
  if (!/[.?!]$/.test(line)) issues.push('End the line with . ? or !');
  const length = state.end - state.start;
  const needed = syllables(line) / speakingRate();
  if (state.buffer && length > 0) {
   const estimate = 'About ' + needed.toFixed(1) + ' s to speak · selection ' + length.toFixed(1) + ' s.';
   issues.unshift(needed > length * 1.25 ? estimate + ' It may sound rushed.' : needed < length * .6 ? estimate + ' The mouth may keep moving after the line, and a voice reference can add invented speech.' : estimate);
  }
 }
 note.hidden = !issues.length;
 note.textContent = issues.join(' ');
}

function reviewTranscriptEdits() {
 try {
  transcript.draft = $('transcriptDraft').value;
  const changes = mapTranscriptChanges(transcript.draft);
  transcript.reviewed = {draft: transcript.draft, changes};
  renderTranscriptChanges();
  window.dispatchEvent(new Event('studio-project-changed'));
 } catch (error) {
  transcript.reviewed = null;
  renderTranscriptChanges();
  $('transcriptEditStatus').textContent = error.message;
 }
}

$('transcriptDraft').addEventListener('input', () => {
 transcript.draft = $('transcriptDraft').value;
 transcript.reviewed = null;
 renderTranscriptChanges();
 window.dispatchEvent(new Event('studio-project-changed'));
});
$('reviewTranscriptEdits').onclick = reviewTranscriptEdits;
$('resetTranscriptDraft').onclick = () => {
 transcript.draft = sourceTranscriptText();
 transcript.reviewed = null;
 $('transcriptDraft').value = transcript.draft;
 renderTranscriptChanges();
 window.dispatchEvent(new Event('studio-project-changed'));
};
$('queueTranscriptEdits').onclick = () => {
 try {
  queueTranscriptEdits();
 } catch (error) {
  $('transcriptEditStatus').textContent = error.message;
 }
};
$('prompt').addEventListener('input', checkLine);
window.addEventListener('studio-selection-changed', () => {
 markTranscript();
 checkLine();
});
window.addEventListener('studio-source-changed', () => {
 transcript.words = [];
 transcript.anchor = null;
 transcript.draft = '';
 transcript.reviewed = null;
 renderTranscript();
});
window.addEventListener('studio-turns-changed', renderTranscriptChanges);
