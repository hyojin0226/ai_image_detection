// Prints configuration presence only. Never print credentials.
const origin = process.env.APP_ORIGIN || 'http://localhost:8000';
for (const [name, label] of [['kakao', 'Kakao'], ['google', 'Google'], ['naver', 'Naver']]) {
  const prefix = name.toUpperCase();
  const id = Boolean(process.env[`${prefix}_CLIENT_ID`]?.trim());
  const secret = Boolean(process.env[`${prefix}_CLIENT_SECRET`]?.trim());
  console.log(`${label}: client ID ${id ? 'set' : 'missing'}, client secret ${secret ? 'set' : 'missing'}`);
  console.log(`  Callback: ${origin}/auth/${name}/callback`);
}
console.log('Presence checks do not verify provider settings or a successful account login.');
