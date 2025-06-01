document.getElementById('detect-btn').addEventListener('click', () => {
  const resultDiv = document.getElementById('result');
  resultDiv.textContent = 'Detecting emotion... Please wait!';

  fetch('/quote')
    .then(response => response.json())
    .then(data => {
      // data contains: { quote, emoji, emotion, gender }
      resultDiv.textContent = `Emotion detected: ${data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1)} ${data.emoji}\n"${data.quote}"`;
    })
    .catch(error => {
      console.error('Error fetching emotion:', error);
      resultDiv.textContent = 'Error detecting emotion. Please try again.';
    });
});
