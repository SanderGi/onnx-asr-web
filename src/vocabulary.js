export class Vocabulary {
  constructor(tokens) {
    this.tokens = tokens;
  }

  isControlToken(token) {
    return token && token.startsWith("<") && token.endsWith(">");
  }

  decodeToken(token) {
    return token.replaceAll("▁", " ");
  }

  decode(tokenIds) {
    return tokenIds
      .map((tokenId) => this.tokens[tokenId])
      .filter((token) => token && !this.isControlToken(token))
      .map((token) => this.decodeToken(token))
      .join("")
      .trim();
  }

  wordTimestamps(tokenIds, tokenFrames, secondsPerFrame) {
    const words = [];
    let current = null;

    const closeCurrent = () => {
      if (!current) {
        return;
      }
      const cleaned = current.text.trim();
      if (cleaned) {
        words.push({
          word: cleaned,
          start: Number((current.startFrame * secondsPerFrame).toFixed(3)),
          end: Number((current.endFrame * secondsPerFrame).toFixed(3)),
        });
      }
      current = null;
    };

    for (let i = 0; i < tokenIds.length; i += 1) {
      const token = this.tokens[tokenIds[i]];
      const frame = tokenFrames[i];
      if (!token || !frame || this.isControlToken(token)) {
        continue;
      }

      const startsNewWord = token.startsWith("▁");
      const piece = this.decodeToken(token);

      if (startsNewWord) {
        closeCurrent();
        current = {
          text: piece.trimStart(),
          startFrame: frame.startFrame,
          endFrame: frame.endFrame,
        };
        continue;
      }

      if (!current) {
        current = {
          text: piece,
          startFrame: frame.startFrame,
          endFrame: frame.endFrame,
        };
      } else {
        current.text += piece;
        current.endFrame = frame.endFrame;
      }
    }

    closeCurrent();
    return words;
  }
}
