/** @type {import('tailwindcss').Config} */

export default {
  darkMode: ["selector", '[data-theme="dark"]'],
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
  	container: {
  		center: true,
  		padding: "2rem",
  		screens: {
  			"2xl": "1400px",
  		},
  	},
  	extend: {
  		fontFamily: {
  			sans: ['Inter', 'system-ui', 'sans-serif'],
  			mono: ['SF Mono', 'Monaco', 'Cascadia Code', 'Roboto Mono', 'Consolas', 'monospace'],
  		},
  		borderRadius: {
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 2px)',
  			sm: 'calc(var(--radius) - 4px)'
  		},
  		colors: {
  			// Colores base del sistema
  			background: 'hsl(var(--background))',
  			foreground: 'hsl(var(--foreground))',
  			card: {
  				DEFAULT: 'hsl(var(--card))',
  				foreground: 'hsl(var(--card-foreground))'
  			},
  			popover: {
  				DEFAULT: 'hsl(var(--popover))',
  				foreground: 'hsl(var(--popover-foreground))'
  			},
  			// Colores actualizados del tema Violet Bloom
  			primary: {
  				DEFAULT: '#7033ff', // Color violeta principal
  				foreground: '#ffffff' // Texto blanco sobre el primary
  			},
  			secondary: {
  				DEFAULT: '#edf0f4', // Gris claro
  				foreground: '#080808' // Texto oscuro sobre el secondary
  			},
  			muted: {
  				DEFAULT: 'hsl(var(--muted))',
  				foreground: 'hsl(var(--muted-foreground))'
  			},
  			accent: {
  				DEFAULT: 'hsl(var(--accent))',
  				foreground: 'hsl(var(--accent-foreground))'
  			},
  			destructive: {
  				DEFAULT: 'hsl(var(--destructive))',
  				foreground: 'hsl(var(--destructive-foreground))'
  			},
  			// Estados de color personalizados
  			success: {
  				DEFAULT: 'hsl(var(--success))',
  				foreground: 'hsl(var(--success-foreground))'
  			},
  			warning: {
  				DEFAULT: 'hsl(var(--warning))',
  				foreground: 'hsl(var(--warning-foreground))'
  			},
  			// Colores por dimensión de TalentAI (actualizados con tonos que complementen el tema)
  			dimension: {
  				1: {
  					DEFAULT: '#7033ff', // Usa el primary como base
  					light: '#9966ff'
  				},
  				2: {
  					DEFAULT: '#5c46ff',
  					light: '#8a7aff'
  				},
  				3: {
  					DEFAULT: '#4d59ff',
  					light: '#7d8cff'
  				},
  				4: {
  					DEFAULT: '#3d6cff',
  					light: '#6f9eff'
  				},
  				5: {
  					DEFAULT: '#2d7fff',
  					light: '#61b1ff'
  				},
  				6: {
  					DEFAULT: '#1d92ff',
  					light: '#53c4ff'
  				},
  				7: {
  					DEFAULT: '#0da5ff',
  					light: '#45d7ff'
  				},
  				8: {
  					DEFAULT: '#00b8ff',
  					light: '#37eaff'
  				}
  			},
  			border: 'hsl(var(--border))',
  			input: 'hsl(var(--input))',
  			ring: 'hsl(var(--ring))'
  		}
  	}
  },
  plugins: [require("tailwindcss-animate")],
};