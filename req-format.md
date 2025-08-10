Example 1: Asking a question (first request, session will be created)
POST https://your-api-url.onrender.com/ask
Headers:
  Content-Type: application/json
Body:
{
    "question": "Is the movie from 2010?"
}

Response:
{
    "answer": "Yes",
    "game_over": false,
    "session_id": "f8d2c6e8-96a5-4bcd-bf8d-6a1f6d4f5a12"
}

Save the "session_id" value from the response and use it for all future requests.

---

Example 2: Asking another question (same session)
POST https://your-api-url.onrender.com/ask
Headers:
  Content-Type: application/json
  X-Session-ID: f8d2c6e8-96a5-4bcd-bf8d-6a1f6d4f5a12
Body:
{
    "question": "Is it an action movie?"
}

---

Example 3: Getting a hint for your current movie
GET https://your-api-url.onrender.com/hint
Headers:
  X-Session-ID: f8d2c6e8-96a5-4bcd-bf8d-6a1f6d4f5a12

