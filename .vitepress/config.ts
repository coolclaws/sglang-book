import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'SGLang 源码解析',
  description: 'RadixAttention 与结构化生成推理运行时深度剖析',
  base: '/',
  lang: 'zh-CN',
  lastUpdated: true,

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/icons/logo.svg' }],
  ],

  themeConfig: {
    logo: '/icons/logo.svg',
    siteTitle: 'SGLang 源码解析',

    nav: [
      { text: '首页', link: '/' },
      { text: '开始阅读', link: '/chapters/01-project-overview' },
      {
        text: '章节导航',
        items: [
          { text: '第一部分：宏观认知', link: '/chapters/01-project-overview' },
          { text: '第二部分：前端语言', link: '/chapters/03-frontend-dsl' },
          { text: '第三部分：RadixAttention', link: '/chapters/06-radix-attention-design' },
          { text: '第四部分：调度与批处理', link: '/chapters/09-scheduler' },
          { text: '第五部分：执行后端', link: '/chapters/12-model-runner' },
          { text: '第六部分：结构化生成', link: '/chapters/16-outlines-integration' },
          { text: '第七部分：API 与部署', link: '/chapters/18-openai-api' },
          { text: '第八部分：生态', link: '/chapters/20-comparison-guide' },
          { text: '附录', link: '/chapters/appendix-a' },
        ]
      }
    ],

    sidebar: [
      {
        text: '第一部分：宏观认知',
        collapsed: false,
        items: [
          { text: '第 1 章：项目概览与设计哲学', link: '/chapters/01-project-overview' },
          { text: '第 2 章：Repo 结构与模块依赖', link: '/chapters/02-repo-structure' },
        ]
      },
      {
        text: '第二部分：前端语言（SGLang DSL）',
        collapsed: false,
        items: [
          { text: '第 3 章：SGLang 前端设计', link: '/chapters/03-frontend-dsl' },
          { text: '第 4 章：Runtime 前端接口', link: '/chapters/04-runtime-frontend' },
          { text: '第 5 章：Sampling 参数与结构化输出', link: '/chapters/05-sampling-params' },
        ]
      },
      {
        text: '第三部分：RadixAttention',
        collapsed: false,
        items: [
          { text: '第 6 章：RadixAttention 设计原理', link: '/chapters/06-radix-attention-design' },
          { text: '第 7 章：RadixCache 实现', link: '/chapters/07-radix-cache-impl' },
          { text: '第 8 章：与 PagedAttention 的对比', link: '/chapters/08-vs-paged-attention' },
        ]
      },
      {
        text: '第四部分：调度与批处理',
        collapsed: false,
        items: [
          { text: '第 9 章：Scheduler 架构', link: '/chapters/09-scheduler' },
          { text: '第 10 章：Continuous Batching', link: '/chapters/10-continuous-batching' },
          { text: '第 11 章：Chunked Prefill 与 PD 分离', link: '/chapters/11-chunked-prefill' },
        ]
      },
      {
        text: '第五部分：执行后端',
        collapsed: false,
        items: [
          { text: '第 12 章：Model Runner', link: '/chapters/12-model-runner' },
          { text: '第 13 章：Attention Backend', link: '/chapters/13-attention-backend' },
          { text: '第 14 章：分布式执行', link: '/chapters/14-distributed' },
          { text: '第 15 章：量化与模型加载', link: '/chapters/15-quantization' },
        ]
      },
      {
        text: '第六部分：结构化生成',
        collapsed: false,
        items: [
          { text: '第 16 章：Outlines 集成', link: '/chapters/16-outlines-integration' },
          { text: '第 17 章：Jump Forward Decoding', link: '/chapters/17-jump-forward' },
        ]
      },
      {
        text: '第七部分：API 与部署',
        collapsed: false,
        items: [
          { text: '第 18 章：OpenAI 兼容 API', link: '/chapters/18-openai-api' },
          { text: '第 19 章：多模态支持', link: '/chapters/19-multimodal' },
        ]
      },
      {
        text: '第八部分：生态',
        collapsed: false,
        items: [
          { text: '第 20 章：对比与选型指南', link: '/chapters/20-comparison-guide' },
        ]
      },
      {
        text: '附录',
        collapsed: false,
        items: [
          { text: '附录 A：推荐阅读路径', link: '/chapters/appendix-a' },
          { text: '附录 B：核心参数速查', link: '/chapters/appendix-b' },
          { text: '附录 C：名词解释', link: '/chapters/appendix-c' },
        ]
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/sgl-project/sglang' }
    ],

    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    docFooter: {
      prev: '上一章',
      next: '下一章'
    },

    editLink: {
      pattern: 'https://github.com/sgl-project/sglang/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    footer: {
      message: '基于 SGLang 开源项目的源码解析',
      copyright: 'Copyright © 2024-present'
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '未找到结果',
            resetButtonTitle: '清除查询',
            footer: { selectText: '选择', navigateText: '导航', closeText: '关闭' }
          }
        }
      }
    },

    lastUpdated: {
      text: '最后更新于'
    }
  }
})
