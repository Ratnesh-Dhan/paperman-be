const express = require("express");
const cors = require("cors");
const app = express();
const axios = require("axios");

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
  try {
    const response = await axios.post(
      "http://localhost:8000/query",
      { query: userQuery },
      {
        headers: {
          "content-type": "application/json",
        },
      }
    );
    console.log(response.data);
    res.json(response.data);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(8080, () => console.log("Server is running on port 8080"));
