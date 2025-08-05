const express = require("express");
const cors = require("cors");
const app = express();

app.use(
  cors({
    origin: "*",
  })
);
app.use(express.json());

app.get("/", (req, res) => {
  res.send("Node server is running");
});

app.post("/ask", async (req, res) => {
  const userQuery = req.body.query;
  console.log("We are inside the ask endpoint");
  console.log(userQuery);

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  try {
    const response = await fetch("http://localhost:8000/query", {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({ query: userQuery }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value, { stream: true });
      process.stdout.write(text); // Use process.stdout.write to avoid newlines
      res.write(text);
      res.flush?.(); // Important for streaming
    }
    res.write("\n[DONE]");
    res.end();
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(8080, () => console.log("Server is running on port 8080"));
