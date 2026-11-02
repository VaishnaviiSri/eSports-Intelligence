const form = document.querySelector('#chat-form');
const questionInput = document.querySelector('#question');
const scopeInput = document.querySelector('#scope');
const sendButton = document.querySelector('#send-button');
const statusText = document.querySelector('#status');
const responsePanel = document.querySelector('#response');
const answerText = document.querySelector('#answer');
const scopeUsed = document.querySelector('#scope-used');
const citationsList = document.querySelector('#citations');
const citationCount = document.querySelector('#citation-count');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  statusText.textContent = '';
  responsePanel.hidden = true;
  citationsList.replaceChildren();
  sendButton.disabled = true;
  sendButton.querySelector('span:first-child').textContent = 'Checking rules…';

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: questionInput.value, scope: scopeInput.value }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || 'The assistant is temporarily unavailable.');
    }

    answerText.textContent = result.answer;
    scopeUsed.textContent = `Scope: ${result.scope_used.replace('_', ' ')}`;
    citationCount.textContent = String(result.citations.length);
    for (const citation of result.citations) {
      const item = document.createElement('li');
      const page = document.createElement('span');
      page.className = 'citation-page';
      page.textContent = `PAGE ${citation.page_number}`;
      const copy = document.createElement('div');
      copy.className = 'citation-copy';
      const heading = document.createElement('strong');
      heading.textContent = citation.section;
      const snippet = document.createElement('p');
      snippet.textContent = citation.content_snippet;
      copy.append(heading, snippet);
      item.append(page, copy);
      citationsList.append(item);
    }
    responsePanel.hidden = false;
  } catch (error) {
    statusText.textContent = error instanceof Error ? error.message : 'Request failed.';
  } finally {
    sendButton.disabled = false;
    sendButton.querySelector('span:first-child').textContent = 'Send question';
  }
});
