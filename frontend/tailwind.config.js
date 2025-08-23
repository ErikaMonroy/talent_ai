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
  			primary: {
  				DEFAULT: 'hsl(var(--primary))',
  				foreground: 'hsl(var(--primary-foreground))'
  			},
  			secondary: {
  				DEFAULT: 'hsl(var(--secondary))',
  				foreground: 'hsl(var(--secondary-foreground))'
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
  			// Colores por dimensión de TalentAI
  			dimension: {
  				1: {
  					DEFAULT: 'hsl(var(--dimension-1))', // Lógico-Matemático
  					light: 'hsl(var(--dimension-1-light))'
  				},
  				2: {
  					DEFAULT: 'hsl(var(--dimension-2))', // Comunicación
  					light: 'hsl(var(--dimension-2-light))'
  				},
  				3: {
  					DEFAULT: 'hsl(var(--dimension-3))', // Ciencias
  					light: 'hsl(var(--dimension-3-light))'
  				},
  				4: {
  					DEFAULT: 'hsl(var(--dimension-4))', // Humanidades
  					light: 'hsl(var(--dimension-4-light))'
  				},
  				5: {
  					DEFAULT: 'hsl(var(--dimension-5))', // Creatividad
  					light: 'hsl(var(--dimension-5-light))'
  				},
  				6: {
  					DEFAULT: 'hsl(var(--dimension-6))', // Liderazgo
  					light: 'hsl(var(--dimension-6-light))'
  				},
  				7: {
  					DEFAULT: 'hsl(var(--dimension-7))', // Pensamiento Crítico
  					light: 'hsl(var(--dimension-7-light))'
  				},
  				8: {
  					DEFAULT: 'hsl(var(--dimension-8))', // Adaptabilidad
  					light: 'hsl(var(--dimension-8-light))'
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
