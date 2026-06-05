export const metadata = {
  title: 'AI Healthcare Chatbot',
  description: 'Real-time AI Health Analytics Assistant',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}