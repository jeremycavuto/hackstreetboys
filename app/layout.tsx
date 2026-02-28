
import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { ClerkProvider } from "@clerk/nextjs" 

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Volt Watch',
  description: 'A comprehensive monitoring and analytics platform for your EV battery, providing real-time insights and performance metrics to help you optimize and scale with confidence.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <ClerkProvider>    
      <html lang="en">
      <body className={inter.className}>{children}</body>
      </html>
    </ClerkProvider>

  )
}
