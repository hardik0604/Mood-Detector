const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const result = document.getElementById('result');
const player = document.getElementById('player');

const moodSongs = {
    happy: "https://www.bensound.com/bensound-music/bensound-happyrock.mp3",
    sad: "https://www.bensound.com/bensound-music/bensound-slowmotion.mp3",
    angry: "https://www.bensound.com/bensound-music/bensound-actionable.mp3",
    neutral: "https://www.bensound.com/bensound-music/bensound-sunny.mp3",
    disgust: "https://www.bensound.com/bensound-music/bensound-creativeminds.mp3",
    fear: "https://www.bensound.com/bensound-music/bensound-anewbeginning.mp3",
    surprised: "https://www.bensound.com/bensound-music/bensound-epic.mp3"
};

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    });

function captureAndSend() {
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const image = canvas.toDataURL('image/jpeg');

    fetch('/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: image })
    })
    .then(res => res.json())
    .then(data => {
        const mood = data.emotion;
        result.innerText = "Detected Mood: " + mood;
        const song = moodSongs[mood] || moodSongs['neutral'];
        player.src = song;
        player.play();
    });
}
