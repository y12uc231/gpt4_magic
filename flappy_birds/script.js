const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

let bird = {
  x: canvas.width / 4,
  y: canvas.height / 2,
  width: 30,
  height: 30,
  dy: 2
};

const marioImg = new Image();
marioImg.src = "mario.png";

let pipes = [];
const pipeWidth = 100;
const pipeGap = 150;
const pipeSpawnRate = 150;

let score = 0;
let frameCount = 0;

function drawBird() {
  ctx.drawImage(marioImg, bird.x, bird.y, bird.width, bird.height);
}

function updateBird() {
  bird.y += bird.dy;
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawPipes() {
  pipes.forEach((pipe) => {
    ctx.fillStyle = "#0f0";
    ctx.fillRect(pipe.x, pipe.y, pipeWidth, pipe.height);
  });
}

function updatePipes() {
  pipes.forEach((pipe) => {
    pipe.x -= 2;
  });

  if (frameCount % pipeSpawnRate === 0) {
    let pipeHeight = Math.random() * (canvas.height - pipeGap * 1.5) + pipeGap * 0.5;
    pipes.push({ x: canvas.width, y: 0, height: pipeHeight });
    pipes.push({ x: canvas.width, y: pipeHeight + pipeGap, height: canvas.height - pipeHeight - pipeGap });
  }

  pipes = pipes.filter((pipe) => pipe.x + pipeWidth > 0);
}

function collisionDetection() {
  if (bird.y + bird.height > canvas.height || bird.y < 0) {
    return true;
  }

  for (let pipe of pipes) {
    if (
      bird.x < pipe.x + pipeWidth &&
      bird.x + bird.width > pipe.x &&
      bird.y < pipe.y + pipe.height &&
      bird.y + bird.height > pipe.y
    ) {
      return true;
    }
  }

  return false;
}

function updateScore() {
  if (frameCount % pipeSpawnRate === 0) {
    score++;
  }
}

function drawScore() {
  ctx.fillStyle = "#000";
  ctx.font = "24px Arial";
  ctx.fillText("Score: " + score, 10, 30);
}

function gameLoop() {
  clearCanvas();
  updateBird();
  updatePipes();
  drawBird();
  drawPipes();
  updateScore();
  drawScore();

  if (collisionDetection()) {
    resetGame();
  } else {
    frameCount++;
    requestAnimationFrame(gameLoop);
  }
}

function resetGame() {
  bird.y = canvas.height / 2;
  bird.dy = 2;
  pipes = [];
  score = 0;
  frameCount = 0;
}

canvas.addEventListener("click", () => {
  bird.dy = -bird.dy;
});



// New event listener for keydown events
window.addEventListener("keydown", (e) => {
  if (e.code === "ArrowUp") {
    bird.dy = -2;
  } else if (e.code === "ArrowDown") {
    bird.dy = 2;
  }
});

// New event listener for keyup events
window.addEventListener("keyup", (e) => {
  if (e.code === "ArrowUp" || e.code === "ArrowDown") {
    bird.dy = 0;
  }
});

gameLoop();