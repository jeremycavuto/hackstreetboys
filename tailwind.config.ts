import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ], plugins: [
  ],
  theme: {
    screens: {
      'mobile': '300px',
      'tablet': '640px',
      // => @media (min-width: 640px) { ... }

      'laptop': '1024px',
      // => @media (min-width: 1024px) { ... }

      'desktop': '1280px',
      // => @media (min-width: 1280px) { ... }
    },
    extend: {

        darkMode: 'class',
       
        root: {
          "base": "fixed top-0 right-0 left-0 z-50 h-modal h-screen overflow-y-auto overflow-x-hidden md:inset-0 md:h-full",
          "show": {
            "on": "flex bg-gray-900 bg-opacity-50 dark:bg-opacity-80",
            "off": "hidden",
            
            
          },
          
          sizes: {
            "sm": "max-w-sm",
            "md": "max-w-md",
            "lg": "max-w-lg",
            "xl": "max-w-xl",
            "2xl": "max-w-2xl",
            "3xl": "max-w-3xl",
            "4xl": "max-w-4xl",
            "5xl": "max-w-5xl",
            "6xl": "max-w-6xl",
            "7xl": "max-w-7xl"
          },
          positions: {
            "top-left": "items-start justify-start",
            "top-center": "items-start justify-center",
            "top-right": "items-start justify-end",
            "center-left": "items-center justify-start",
            "center": "items-center justify-center",
            "center-right": "items-center justify-end",
            "bottom-right": "items-end justify-end",
            "bottom-center": "items-end justify-center",
            "bottom-left": "items-end justify-start"
          }
        },
        content: {
          "base": "relative h-full w-full p-4 md:h-auto",
          "inner": "relative rounded-lg bg-white shadow dark:bg-gray-700 flex flex-col max-h-[90vh]"
        },
        body: {
          "base": "p-6 flex-1 overflow-auto",
          "popup": "pt-0"
        },
        header: {
          "base": "flex items-start justify-between rounded-t dark:border-gray-600 border-b p-5",
          "popup": "p-2 border-b-0",
          "title": "text-xl font-medium text-gray-900 dark:text-white",
          "close": {
            "base": "ml-auto inline-flex items-center rounded-lg bg-transparent p-1.5 text-sm text-gray-400 hover:bg-gray-200 hover:text-gray-900 dark:hover:bg-gray-600 dark:hover:text-white",
            "icon": "h-5 w-5"
          }
        },
        footer: {
          "base": "flex items-center space-x-2 rounded-b border-gray-200 p-6 dark:border-gray-600",
          "popup": "border-t"
        }
      ,
      colors: {
        transparent: 'transparent',
        current: 'currentColor',
        'white': '#ffffff',
        'purple': '#3f3cbb',
        'midnight': '#121063',
        'metal': '#565584',
        'tahiti': '#3ab7bf',
        'silver': '#ecebff',
        'bubble-gum': '#ff77e9',
        'bermuda': '#78dcca',
        'chemisphere': "#E62020",
        "tickgreen" :"#00D300",
        "whatsapp-green": "#0FB73B",
        "black": "#252525",
        "wave": "#1B003D"

      },
      
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic':
          'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
          "background1" : "url(/bg.svg)",
          "background-mobile" : "url(/chemisphere-bg-mobile.svg)",
          "background-hero" : "url(/background-hero.svg)",
          "bg-wave": "url(/wave.svg)",
          "bg-banner-wave": "url(/banner-wave.svg)",
          "bg-bg-hero": "url(/bg-hero.svg)",
          "bg-banner-JEE": "url(/bg-banner-JEE.svg)",
          "bgg": "url(/Hero.svg)"
      },
      
        
    }
  
  }}
export default config